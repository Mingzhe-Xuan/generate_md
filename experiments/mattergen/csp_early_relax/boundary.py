"""Observe exact completed native PC boundaries without altering sampling."""


class Boundary:
    def __init__(self,module,torch,capture):
        self.module=module;self.torch=torch;self.capture=capture
        self.original=module._mask_replace;self.calls=0

    def __enter__(self):
        self.module._mask_replace=self
        return self

    def __exit__(self,*args):
        self.module._mask_replace=self.original

    def __call__(self,*args,**kwargs):
        result=self.original(*args,**kwargs);self.calls+=1
        if self.calls in (1600,2000):
            cpu=self.torch.get_rng_state().clone()
            cuda=self.torch.cuda.get_rng_state().clone() if self.torch.cuda.is_available() else None
            self.capture(self.calls//2,*result)
            if not self.torch.equal(cpu,self.torch.get_rng_state()) or (cuda is not None and not self.torch.equal(cuda,self.torch.cuda.get_rng_state())):
                raise RuntimeError('Boundary capture changed RNG.')
        return result
