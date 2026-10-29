"""RNG-neutral sparse observation of the unchanged official PC sampler."""


class Observer:
    def __init__(self,sampler,steps,capture,torch):
        if sampler._n_steps_corrector!=1 or not sampler._correctors:
            raise ValueError('This observer expects official one-corrector/one-predictor schedule.')
        self.original=sampler._score_fn
        self.steps=set(steps)
        self.capture=capture
        self.torch=torch
        self.calls=0
        self.snapshots=0

    def __call__(self,x,t):
        score=self.original(x,t)
        step,within=divmod(self.calls,2)
        self.calls+=1
        stage='prior' if step==0 and within==0 else 'predictor'
        if stage=='prior' or (within==1 and step in self.steps):
            cpu_rng=self.torch.get_rng_state().clone()
            cuda_rng=self.torch.cuda.get_rng_state().clone() if self.torch.cuda.is_available() else None
            self.capture(x,t,score,stage,step)
            if not self.torch.equal(cpu_rng,self.torch.get_rng_state()):
                raise RuntimeError('Observer changed CPU RNG state.')
            if cuda_rng is not None and not self.torch.equal(cuda_rng,self.torch.cuda.get_rng_state()):
                raise RuntimeError('Observer changed CUDA RNG state.')
            self.snapshots+=1
        return score
