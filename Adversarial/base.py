from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
'''
Base.py
As the base class for all Adversarial Attacks.
'''

class Attack(ABC):

    def __init__(self,transform:  None,device: None,) -> None:
        super().__init__()
        # If transform is None -> Just original X
        self.transform = transform if transform else lambda x: x

        # Set device to given or defaults to cuda if available.
        # Use CPU to make the code useful on my fucking mac withouot cuda
        is_cuda = torch.cuda.is_available()
        self.device = device if device else torch.device("cuda" if is_cuda else "cpu")

    # Make the class could be called as function
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    # print(class) will show all the args
    def __repr__(self) -> str:
        name = self.__class__.__name__

        def repr_map(k, v):
            if isinstance(v, float):
                return f"{k}={v:.3f}"
            if k in ["model", "transform"]:
                return f"{k}={v.__class__.__name__}"
            if isinstance(v, torch.Tensor):
                return f"{k}={v.shape}"
            return f"{k}={v}"

        args = ", ".join(repr_map(k, v) for k, v in self.__dict__.items())
        return f"{name}({args})"

    # Wait for implement
    @abstractmethod
    def forward(self, *args: Any, **kwds: Any):
        pass
