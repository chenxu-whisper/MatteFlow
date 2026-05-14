"""模型可用性检查工具"""

import importlib
import importlib.util
from typing import Dict, List, Tuple

from .model_paths import model_file, models_root, resolve_snapshot_model_dir


class ModelChecker:
    """检查 AI 模型是否可用"""
    
    def __init__(self):
        self.cache_dir = models_root()
        self.matteflow_dir = models_root()
        
    def check_all_models(self) -> Dict[str, Dict]:
        """检查所有模型状态"""
        results = {}
        
        # 1. GVM
        results["gvm"] = self._check_gvm()
        
        # 2. MatAnyone2
        results["matanyone2"] = self._check_matanyone2()
        
        # 3. SAM2
        results["sam2"] = self._check_sam2()
        
        # 4. BiRefNet
        results["birefnet"] = self._check_birefnet()
        
        # 5. CorridorKey
        results["corridorkey"] = self._check_corridorkey()
        
        # 6. RVM
        results["rvm"] = self._check_rvm()
        
        # 7. rembg
        results["rembg"] = self._check_rembg()
        
        return results
    
    def _check_gvm(self) -> Dict:
        """检查 GVM"""
        model_dir = resolve_snapshot_model_dir(
            self.cache_dir,
            "geyongtao/gvm",
            ("unet", "vae", "scheduler"),
        )
        model_exists = model_dir is not None
        cuda_available = False
        try:
            torch = importlib.import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        if not model_exists:
            reason = "模型仓库不存在或需要授权"
        elif not cuda_available:
            reason = "GVM 仅支持 CUDA GPU"
        else:
            reason = None

        return {
            "name": "GVM (Generative Video Matting)",
            "available": model_exists and cuda_available,
            "path": str(model_dir or (self.cache_dir / "models--geyongtao--gvm")),
            "size": "~6 GB",
            "auto_download": False,
            "reason": reason,
        }
    
    def _check_matanyone2(self) -> Dict:
        """检查 MatAnyone2"""
        model_path = self.matteflow_dir / "matanyone2.pth"
        available = model_path.exists()
        reason = None

        if not available:
            reason = "需要手动下载"
        else:
            try:
                importlib.import_module("matteflow.vendor.matanyone2_module.wrapper")
            except Exception:
                available = False
                reason = "MatAnyone2 vendored runtime 不可导入"

        return {
            "name": "MatAnyone2",
            "available": available,
            "path": str(model_path),
            "size": "~135 MB",
            "auto_download": False,
            "reason": reason
        }
    
    def _check_sam2(self) -> Dict:
        """检查 SAM2"""
        model_path = self.cache_dir / "models--facebook--sam2.1-hiera-base-plus"
        return {
            "name": "SAM2 (Segment Anything 2)",
            "available": model_path.exists(),
            "path": str(model_path),
            "size": "~324 MB",
            "auto_download": True,
            "reason": None
        }
    
    def _check_birefnet(self) -> Dict:
        """检查 BiRefNet"""
        model_path = self.cache_dir / "models--ZhengPeng7--BiRefNet"
        return {
            "name": "BiRefNet",
            "available": model_path.exists(),
            "path": str(model_path),
            "size": "~940 MB",
            "auto_download": True,
            "reason": None
        }
    
    def _check_corridorkey(self) -> Dict:
        """检查 CorridorKey"""
        model_path = model_file("corridorkey.pth")
        available = model_path.exists()
        reason = None

        if not available:
            reason = "需要手动下载"
        else:
            try:
                importlib.import_module("matteflow.vendor.corridorkey_module.inference_engine")
            except Exception:
                available = False
                reason = "CorridorKey vendored runtime 不可导入"

        return {
            "name": "CorridorKey",
            "available": available,
            "path": str(model_path),
            "size": "~383 MB",
            "auto_download": False,
            "reason": reason
        }
    
    def _check_rvm(self) -> Dict:
        """检查 RVM"""
        model_path = model_file("rvm_mobilenetv3.pth")
        return {
            "name": "RVM (Robust Video Matting)",
            "available": model_path.exists(),
            "path": str(model_path),
            "size": "~15 MB",
            "auto_download": True,
            "reason": None
        }
    
    def _check_rembg(self) -> Dict:
        """检查 rembg"""
        available = importlib.util.find_spec("rembg") is not None
        reason = None if available else "未安装 rembg 包"
        
        return {
            "name": "rembg",
            "available": available,
            "path": "pip package",
            "size": "~90 MB",
            "auto_download": True,
            "reason": reason
        }
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        results = self.check_all_models()
        return [name for name, info in results.items() if info["available"]]
    
    def get_ui_choices(self) -> List[Tuple[str, str]]:
        """获取 UI 选项列表"""
        results = self.check_all_models()
        choices = []
        
        # 按优先级排序
        priority_order = ["gvm", "matanyone2", "corridorkey", "sam2", "birefnet", "rvm", "rembg"]
        
        for model_name in priority_order:
            if model_name in results:
                info = results[model_name]
                if info["available"]:
                    # 可用模型
                    label = f"✅ {info['name']}"
                    choices.append((label, model_name))
                else:
                    # 不可用模型（禁用）
                    label = f"❌ {info['name']} ({info['reason'] or '未安装'})"
                    # 不添加到选项中，或添加为禁用状态
        
        # 添加传统算法
        choices.append(("📐 传统算法", "traditional"))
        
        return choices
    
    def print_status(self):
        """打印模型状态"""
        print("=" * 70)
        print("[STATUS] AI 模型状态检查")
        print("=" * 70)
        
        results = self.check_all_models()
        
        for model_name, info in results.items():
            status = "[OK] 可用" if info["available"] else "[FAIL] 不可用"
            print(f"\n{info['name']}")
            print(f"  状态: {status}")
            print(f"  大小: {info['size']}")
            print(f"  路径: {info['path']}")
            if info["reason"]:
                print(f"  原因: {info['reason']}")
            if info["auto_download"]:
                print(f"  自动下载: 支持")
        
        print("\n" + "=" * 70)
        available = self.get_available_models()
        print(f"可用模型: {len(available)}/7")
        print(f"列表: {', '.join(available) if available else '无'}")
        print("=" * 70)


if __name__ == "__main__":
    checker = ModelChecker()
    checker.print_status()
