"""フラットな Transformer デコーダ（E1 + D2 + トークンごとリークヘッド）。

トークン = 測定（raw, 576）または検出器（soft/hard, 504）。位置は
量子ビット埋め込み(72) + ラウンド埋め込み(8) の和で、3 表現で共通。
  - 論理ヘッド：学習する k=12 個のクエリがエンコーダ出力に cross-attention → 各 1 logit
  - リークヘッド：トークンごと Linear(d→1)。`leak_prob` で確率を取り出す（BP-OSD との
    ハイブリッド評価に使う）
  - single_shot=True：attention を同一ラウンド内に制限するマスク。構造とパラメータ数は
    同一で、時間方向の処理だけを無効化する対照
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SyndromeTransformer(nn.Module):
    def __init__(self, d_in: int, qubit_idx, round_idx, *, num_qubits: int = 72,
                 num_rounds: int = 8, k: int = 12, d: int = 128, layers: int = 6,
                 heads: int = 4, ff: int = 512, dropout: float = 0.0, single_shot: bool = False):
        super().__init__()
        qubit_idx = torch.as_tensor(qubit_idx, dtype=torch.long)
        round_idx = torch.as_tensor(round_idx, dtype=torch.long)
        if qubit_idx.shape != round_idx.shape or qubit_idx.ndim != 1:
            raise ValueError("qubit_idx / round_idx は同じ長さの 1 次元")
        self.register_buffer("qubit_idx", qubit_idx)
        self.register_buffer("round_idx", round_idx)
        # AUDIT(2026-09-16) 【重要・交絡】single_shot はエンコーダの attention を同一ラウンド内に閉じるが、
        #   論理ヘッドは 12 クエリの cross-attention 1 層（重み付き平均）しか持たない。したがって単発対照は
        #   「同一量子ビットの連続ラウンドの XOR（=検出器）」も「時空 H による復号」も作れない。
        #   ただし補助測定はリセット後の測定なので各ラウンドの値は累積シンドロームそのものであり（特に無雑音の
        #   最終ラウンド 7）、ラウンド単位の復号は可能。実際 r4 の論理 BCE は step 6500 で 0.63 と下がり始めている
        #   （訂正：当初「偶然水準に張り付く」と書いたが、それは 5000 step の LER だけを見た誤り）。
        #   問題は残る：① vs ④ の差には「時空復号（検出器差分）の有無」と「リーク持続性の推論の有無」が混ざる。
        #   修正案（コードは変えず評価の使い方で対応）：
        #     (a) ④ は evaluate.py --hybrid（リークヘッド → p_S → BP-OSD）で ① の hybrid と比べる。
        #         両者とも復号器は BP-OSD なので、差はリーク推論の時間方向の有無だけになる。
        #     (b) end-to-end の機構証明は p_seep=1（i.i.d.）掃引で行う（モデルは系列のまま、データ側で持続性を消す）。
        #   もし end-to-end の単発対照がどうしても要るなら、リークヘッド用の単発エンコーダと復号用の系列エンコーダを
        #   分け、復号側には検出器レベルの soft 値だけを渡す 2 段構成にする（未実装）：
        #   # self.leak_encoder = nn.TransformerEncoder(layer_ss, layers)   # 同一ラウンド内マスク（リークヘッド専用）
        #   # self.dec_encoder  = nn.TransformerEncoder(layer_full, layers) # 全ラウンド attention（復号専用、入力は boxplus 済み検出器）
        # True = attention 禁止。単発対照は「別ラウンドを見ない」
        mask = round_idx[:, None] != round_idx[None, :] if single_shot else None
        self.register_buffer("attn_mask", mask, persistent=False)
        self.single_shot = single_shot
        self.k = k

        self.input_proj = nn.Linear(d_in, d)
        self.qubit_emb = nn.Embedding(num_qubits, d)
        self.round_emb = nn.Embedding(num_rounds, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d)

        self.logical_queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(d)
        self.logical_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.leak_head = nn.Linear(d, 1)

    @property
    def num_tokens(self) -> int:
        return int(self.qubit_idx.shape[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.num_tokens:
            raise ValueError(f"トークン数 {x.shape[1]} が仕様 {self.num_tokens} と違う")
        h = self.input_proj(x) + self.qubit_emb(self.qubit_idx) + self.round_emb(self.round_idx)
        h = self.encoder(h, mask=self.attn_mask)
        return self.final_norm(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, T, d_in) → (logical_logits (B, k), leak_logits (B, T))"""
        h = self.encode(x)
        q = self.logical_queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        a, _ = self.cross_attn(q, h, h, need_weights=False)
        logical = self.logical_mlp(self.query_norm(q + a)).squeeze(-1)
        leak = self.leak_head(h).squeeze(-1)
        return logical, leak

    @torch.no_grad()
    def leak_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x)[1])


def losses(logical_logits, leak_logits, logical, leak, lam: float = 1.0):
    """L = BCE(論理) + λ BCE(リーク)。両項とも平均。"""
    l_logical = F.binary_cross_entropy_with_logits(logical_logits, logical)
    l_leak = F.binary_cross_entropy_with_logits(leak_logits, leak)
    return l_logical + lam * l_leak, l_logical, l_leak


@torch.no_grad()
def logical_failures(logical_logits: torch.Tensor, logical: torch.Tensor) -> torch.Tensor:
    """(B,) bool：12 ビットのどれかを外したショット。"""
    return ((logical_logits > 0) != (logical > 0.5)).any(dim=1)
