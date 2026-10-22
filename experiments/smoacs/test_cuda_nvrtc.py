from types import SimpleNamespace
from cuda_nvrtc import NvrtcCompiler

def test_preserves_kernel_source_and_original_options(tmp_path):
    calls = []
    def raw(*args, **kwargs):
        calls.append((args, kwargs))
        return "kernel"
    cuda = object()
    cupy = SimpleNamespace(RawKernel=raw, cuda=cuda)
    compiler = NvrtcCompiler(cupy, tmp_path)
    code = 'extern "C" __global__ void original() {}'
    assert compiler.RawKernel(code, "original", ("-dc", "--std=c++11"), jitify=True) == "kernel"
    args, kwargs = calls[0]
    assert args == (code, "original", ("-dc", "--std=c++11", "-I"+str(tmp_path.resolve())))
    assert kwargs == {"jitify": False}
    assert compiler.cuda is cuda
    assert cupy.RawKernel is raw
