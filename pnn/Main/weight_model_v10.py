import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTROL_CKPT_PATH = os.environ.get(
    "PNN_CONTROL_CKPT",
    os.path.join(PROJECT_ROOT, "checkpoints", "pnn_control.pth"),
)


import torch
from torch import nn
import torch.nn.functional as F

class WeightNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_tasks: int = 8,  
        temperature: float = 1.0,
        num_transformer_layers: int = 2,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.1,
        use_prior_context: bool = False,
        prior_context_mode: str = "log",
        initial_refine_gate: float = 0.01,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        self.use_prior_context = bool(use_prior_context)
        self.prior_context_mode = str(prior_context_mode).lower()
        if self.prior_context_mode not in {"log", "raw", "prob"}:
            raise ValueError("prior_context_mode must be one of: log, raw, prob")

        # ===== Encoders =====
        self.ego_encoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.ped_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.veh_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

        # ===== Transformer Fusion =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.attn_fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # ===== Residual MLP =====
        self.res_fc1 = nn.Linear(embed_dim, embed_dim)
        self.res_fc2 = nn.Linear(embed_dim, embed_dim)

        # ===== Weight Decoder =====
        decoder_input_dim = embed_dim + (num_tasks if self.use_prior_context else 0)
        self.weight_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, num_tasks)  
        )
        self.refine_gate_decoder = nn.Linear(decoder_input_dim, 1)

        self._init_decoder()
        initial_refine_gate = min(max(float(initial_refine_gate), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            nn.init.zeros_(self.refine_gate_decoder.weight)
            self.refine_gate_decoder.bias.fill_(
                torch.logit(torch.tensor(initial_refine_gate)).item()
            )

    def _init_decoder(self) -> None:
        for m in self.weight_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_prior_context(
        self,
        prior_weights: torch.Tensor = None,
        prior_log_weights: torch.Tensor = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if prior_log_weights is None and prior_weights is None:
            raise ValueError("prior_weights or prior_log_weights is required when use_prior_context=True")

        if prior_log_weights is not None:
            prior_context = prior_log_weights
        elif self.prior_context_mode == "log":
            prior_context = prior_weights.clamp_min(1e-8).log()
        elif self.prior_context_mode == "prob":
            prior_context = prior_weights.clamp_min(1e-8)
            prior_context = prior_context / prior_context.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            prior_context = prior_weights

        prior_context = prior_context.to(device=device, dtype=dtype)
        if prior_context.dim() != 2 or prior_context.shape[-1] != self.num_tasks:
            raise ValueError(
                f"prior context must have shape [B,{self.num_tasks}], got {tuple(prior_context.shape)}"
            )
        return torch.nan_to_num(prior_context, nan=0.0, posinf=20.0, neginf=-20.0)

    def forward(
        self,
        ego_state: torch.Tensor,
        ped_states: torch.Tensor,
        veh_states: torch.Tensor,
        lane_points: torch.Tensor,
        ped_mask: torch.Tensor = None,
        veh_mask: torch.Tensor = None,
        prior_weights: torch.Tensor = None,
        prior_log_weights: torch.Tensor = None,
        return_logits: bool = False,
    ):
        B = ego_state.size(0)
        Np = ped_states.size(1)
        Nv = veh_states.size(1)

        # ===== Encoding =====
        ego_feat = self.ego_encoder(ego_state).unsqueeze(1)
        ped_feat = self.ped_encoder(ped_states)
        veh_feat = self.veh_encoder(veh_states)
        map_tokens = lane_points.view(B, -1, 2)
        map_feat = self.map_encoder(map_tokens)

        # ===== Feature Fusion =====
        all_feat = torch.cat([ego_feat, ped_feat, veh_feat, map_feat], dim=1)

        total_len = all_feat.shape[1]
        key_padding_mask = torch.zeros(
            B, total_len, dtype=torch.bool, device=ego_state.device
        )
        if ped_mask is not None:
            key_padding_mask[:, 1:1 + Np] = ~ped_mask.bool()
        if veh_mask is not None:
            key_padding_mask[:, 1 + Np:1 + Np + Nv] = ~veh_mask.bool()

        fused_feat = self.attn_fusion(
            all_feat, src_key_padding_mask=key_padding_mask
        )
        fused_feat = fused_feat + F.relu(
            self.res_fc2(F.relu(self.res_fc1(fused_feat)))
        )

        # 使用 ego token 作为全局场景特征
        scene_feat = fused_feat[:, 0]
        if self.use_prior_context:
            prior_context = self._build_prior_context(
                prior_weights=prior_weights,
                prior_log_weights=prior_log_weights,
                device=scene_feat.device,
                dtype=scene_feat.dtype,
            )
            scene_feat = torch.cat([scene_feat, prior_context], dim=-1)

        # ===== Weight Prediction =====
        logits = self.weight_decoder(scene_feat)
        weights = F.softmax(logits / self.temperature, dim=-1)
        refine_gate = torch.sigmoid(self.refine_gate_decoder(scene_feat))

        if return_logits:
            return weights, logits, refine_gate
        return weights


def load_control_encoder_to_weightnet(
    weight_net: nn.Module,
    ckpt_path: str = CONTROL_CKPT_PATH,
    map_location: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    只从 ckpt['neural_net'] 迁移 WeightNet 前半部分同构参数：
        ego_encoder
        ped_encoder
        veh_encoder
        map_encoder
        attn_fusion
        res_fc1
        res_fc2

    不迁移：
        ego_output / ped_output / veh_output
        weight_decoder
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location)

    if "neural_net" not in ckpt:
        raise KeyError("checkpoint 中不存在 'neural_net' 键")

    source_state = ckpt["neural_net"]
    if not isinstance(source_state, dict):
        raise TypeError("ckpt['neural_net'] 不是 state_dict")

    target_state = weight_net.state_dict()

    transferable_prefixes = [
        "ego_encoder",
        "ped_encoder",
        "veh_encoder",
        "map_encoder",
        "attn_fusion",
        "res_fc1",
        "res_fc2",
    ]

    loaded = []
    skipped_not_found = []
    skipped_shape_mismatch = []

    new_state = target_state.copy()

    for target_key in target_state.keys():
        if not any(target_key.startswith(prefix) for prefix in transferable_prefixes):
            continue

        if target_key not in source_state:
            skipped_not_found.append(target_key)
            continue

        if source_state[target_key].shape != target_state[target_key].shape:
            skipped_shape_mismatch.append(
                f"{target_key}: source={tuple(source_state[target_key].shape)}, target={tuple(target_state[target_key].shape)}"
            )
            continue

        new_state[target_key] = source_state[target_key]
        loaded.append(target_key)

    weight_net.load_state_dict(new_state, strict=False)

    report = {
        "loaded": loaded,
        "skipped_not_found": skipped_not_found,
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }

    if verbose:
        print("=" * 100)
        print("Partial load report: ControlNet -> WeightNet")
        print("checkpoint:", ckpt_path)
        print("loaded:", len(loaded))
        print("skipped_not_found:", len(skipped_not_found))
        print("skipped_shape_mismatch:", len(skipped_shape_mismatch))

        if loaded:
            print("\n[loaded keys]")
            for k in loaded:
                print(" ", k)

        if skipped_not_found:
            print("\n[skipped_not_found]")
            for k in skipped_not_found:
                print(" ", k)

        if skipped_shape_mismatch:
            print("\n[skipped_shape_mismatch]")
            for k in skipped_shape_mismatch:
                print(" ", k)
        print("=" * 100)

    return report


def build_optimizer(
    model: nn.Module,
    encoder_lr: float = 1e-4,
    decoder_lr: float = 5e-4,
    weight_decay: float = 1e-4,
):
    encoder_params = (
        list(model.ego_encoder.parameters()) +
        list(model.ped_encoder.parameters()) +
        list(model.veh_encoder.parameters()) +
        list(model.map_encoder.parameters()) +
        list(model.attn_fusion.parameters()) +
        list(model.res_fc1.parameters()) +
        list(model.res_fc2.parameters())
    )

    decoder_params = list(model.weight_decoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def freeze_pretrained_part(model: nn.Module) -> None:
    modules = [
        model.ego_encoder,
        model.ped_encoder,
        model.veh_encoder,
        model.map_encoder,
        model.attn_fusion,
        model.res_fc1,
        model.res_fc2,
    ]
    for module in modules:
        for p in module.parameters():
            p.requires_grad = False


def unfreeze_pretrained_part(model: nn.Module) -> None:
    modules = [
        model.ego_encoder,
        model.ped_encoder,
        model.veh_encoder,
        model.map_encoder,
        model.attn_fusion,
        model.res_fc1,
        model.res_fc2,
    ]
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def entropy_reg(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(weights * torch.log(weights + eps)).sum(dim=-1).mean()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WeightNet(
        embed_dim=128,
        num_heads=4,
        num_tasks=8,
        temperature=1.0
    )

    # 1. 加载原控制网络的前半部分参数
    load_control_encoder_to_weightnet(
        weight_net=model,
        ckpt_path=CONTROL_CKPT_PATH,
        map_location="cpu",
        verbose=True
    )

    model = model.to(device)

    # 2. 假数据检查
    #B, Np, Nv, Nl, P = 2, 10, 10, 10, 20
    #ego_state = torch.randn(B, 10, device=device)
    #ped_states = torch.randn(B, Np, 6, device=device)
    #veh_states = torch.randn(B, Nv, 6, device=device)
    #lane_points = torch.randn(B, Nl, P, 2, device=device)
    #ped_mask = torch.zeros(B, Np, dtype=torch.bool, device=device)
    #veh_mask = torch.zeros(B, Nv, dtype=torch.bool, device=device)

    #weights, logits = model(
     #   ego_state=ego_state,
     #   ped_states=ped_states,
     #   veh_states=veh_states,
     #   lane_points=lane_points,
     #   ped_mask=ped_mask,
     #   veh_mask=veh_mask,
      #  return_logits=True
    #)

    #print("weights.shape:", weights.shape)
    #print("weights.sum(dim=-1):", weights.sum(dim=-1))

    # 3. 优化器
    optimizer = build_optimizer(
        model,
        encoder_lr=1e-4,
        decoder_lr=5e-4
    )

    # 4. 最小训练步示意
    dummy_task_loss = torch.rand(B, 6, device=device)
    loss = (weights * dummy_task_loss).sum(dim=-1).mean() - 0.01 * entropy_reg(weights)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("dummy step done, loss =", float(loss.detach().cpu()))
