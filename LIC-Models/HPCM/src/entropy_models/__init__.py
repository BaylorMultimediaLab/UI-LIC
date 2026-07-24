try:
    from ._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
except ImportError:
    from _CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

try:
    from .unbounded_ans import ubransEncoder, ubransDecoder
except ImportError:
    from unbounded_ans import ubransEncoder, ubransDecoder