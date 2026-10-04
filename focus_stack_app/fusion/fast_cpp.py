"""C++ v3 pixel kernels; OpenCV supplies its existing native image operators."""
import ctypes as C
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

LIBRARY = Path(__file__).with_name('fast_core.dll')
if not LIBRARY.is_file():
    raise RuntimeError('C++ v3 DLL is missing; run build_cpp.py with the provided compiler')
_dll = C.CDLL(str(LIBRARY))
_calls = Counter()
P, I, N, F, U = C.c_void_p, C.c_int, C.c_int64, C.c_float, C.c_uint16

def bind(name, args, result=None):
    fn = getattr(_dll, name)
    fn.argtypes, fn.restype = args, result
    return fn

_abi = bind('fast_core_abi', [], I)
if _abi() != 3:
    raise RuntimeError('C++ v3 DLL ABI mismatch')
_chroma = bind('rgb_chroma', [P,P,I,I,N])
_independent = bind('independent_detail', [P,P,P,F,P,N])
_features = bind('focus_fields', [P]*6+[F]+[P]*3+[N])
_winners = bind('proxy_winners', [P]*5+[U]+[P]*8+[N])
_preserve = bind('preserve_detail', [P]*7+[N])
_neutral_filter = bind('neutral_seed_filter', [P]*7+[N])
_neutral_update = bind('neutral_update', [P]*6+[U,N])
_neutral_apply = bind('neutral_apply', [P]*6+[N])
_inputs = bind('support_inputs', [P]*4+[N])
_support = bind('nearest_support', [P]*5+[F,P,N], I)
_energy = bind('contour_energy', [P]*5+[N])
_normalize = bind('normalize_field', [P,P,N])
_rank = bind('native_rank', [P]*7+[I]*5+[U,I])
_capture = bind('native_capture', [P]*4+[I]*6)
_select = bind('native_select', [P]*8+[I]*5+[U,P])
_blend = bind('native_blend', [P]*4+[F]+[I]*5)
_create = bind('render_create', [P,P,N], P)
_destroy = bind('render_destroy', [P])
_seams = bind('render_seams', [P], N)
_owner = bind('render_owner', [P,U,P], N)
_add = bind('render_add', [P,U,P,P,P,P])
_finish = bind('render_finish', [P,P], N)

def ptr(a):
    if not a.flags.c_contiguous:
        raise ValueError('C++ v3 buffer must be contiguous')
    return a.ctypes.data

def inputs(*arrays):
    return [ptr(a) for a in arrays]

def runtime_info():
    return {'library': str(LIBRARY), 'abi': int(_abi()), 'calls': dict(_calls),
            'implementation': 'C++17 DLL + OpenCV native operators'}

def chroma(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 image required')
    if rgb.strides[1:] != (3,1) or rgb.strides[0] <= 0:
        rgb = np.ascontiguousarray(rgb)
    result = np.empty(rgb.shape[:2], np.uint8)
    _chroma(rgb.ctypes.data,ptr(result),*result.shape,rgb.strides[0])
    _calls['chroma'] += 1
    return result

def independent(detail, broad, gradient, floor):
    out = np.empty(detail.shape, bool)
    _independent(*inputs(detail,broad,gradient),float(floor),ptr(out),out.size)
    _calls['independent_detail'] += 1
    return out

def focus_fields(xx,yy,xy,gradient,variance,detail,floor):
    base,sharp = (np.empty(detail.shape,np.float32) for _ in range(2))
    texture = np.empty(detail.shape,bool)
    _features(*inputs(xx,yy,xy,gradient,variance,detail),float(floor),
              *inputs(base,sharp,texture),detail.size)
    _calls['focus_fields'] += 1
    return base,sharp,texture

def proxy_winners(valid,evidence,texture,score,detail,index,best,labels,
                  detail_best,detail_owner,local_best,local_owner,textured):
    local_better = np.empty(valid.shape,bool)
    _winners(*inputs(valid,evidence,texture,score,detail),index,
             *inputs(best,labels,detail_best,detail_owner,local_best,local_owner,textured,local_better),valid.size)
    _calls['proxy_winners'] += 1
    return local_better

def preserve_detail(labels,propagated,propagated_detail,local_best,local_owner,active):
    result,independent = labels.copy(),np.empty(labels.shape,bool)
    _preserve(*inputs(result,propagated,propagated_detail,local_best,local_owner,active,independent),labels.size)
    _calls['preserve_detail'] += 1
    return result,independent

def nearest_support(strength,edge,radius):
    strength = np.ascontiguousarray(strength,dtype=np.float32)
    edge = np.ascontiguousarray(edge != 0)
    if not np.any(edge):
        return np.zeros_like(strength)
    present,masked = (np.empty(strength.shape,np.float32) for _ in range(2))
    _inputs(*inputs(strength,edge,present,masked),strength.size)
    density = cv2.GaussianBlur(present,(0,0),2.0)
    averaged = cv2.GaussianBlur(masked,(0,0),2.0)
    distance,nearest = cv2.distanceTransformWithLabels(
        np.uint8(~edge),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    out = np.empty_like(strength)
    if _support(*inputs(edge,density,averaged,distance,nearest),float(radius),ptr(out),out.size):
        raise MemoryError('C++ nearest edge support allocation failed')
    _calls['nearest_support'] += 1
    return out

def neutral_filter(positions,details,local_best,chroma,foreground,material):
    use = np.empty(positions.shape,bool)
    _neutral_filter(*inputs(positions,details,local_best,chroma,foreground,material,use),positions.size)
    _calls['neutral_seed_filter'] += 1
    return use

def neutral_update(targets,material,strength,best,owner,chroma,index):
    _neutral_update(*inputs(targets,material,strength,best,owner,chroma),index,best.size)
    _calls['neutral_update'] += 1

def neutral_apply(labels,owner,best,chroma,protected):
    result = labels.copy()
    counts = np.empty(2,np.int64)
    _neutral_apply(*inputs(result,owner,best,chroma,protected,counts),labels.size)
    _calls['neutral_apply'] += 1
    return result,int(counts[0]),int(counts[1])

def contour_energy(gradient,detail,seeds):
    present,weighted = (np.empty(gradient.shape,np.float32) for _ in range(2))
    _energy(*inputs(gradient,detail,seeds,present,weighted),gradient.size)
    density = cv2.GaussianBlur(present,(0,0),8.0)
    coherent = cv2.GaussianBlur(weighted,(0,0),8.0)
    _normalize(*inputs(coherent,density),gradient.size)
    _calls['contour_energy'] += 1
    return coherent

def geometry(tile, full_w):
    return [tile['y1']-tile['y'],tile['x1']-tile['x'],tile['x'],tile['y'],full_w]

def rank(tile, valid, score, index, reference):
    better = np.empty(tile['best'].shape,bool)
    score = np.ascontiguousarray(score,dtype=np.float32)
    _rank(*inputs(tile['best'],tile['owner'],tile['reference_score'],tile['mask'],valid,score,better),
          *geometry(tile,valid.shape[1]),index,int(reference))
    _calls['native_rank'] += 1
    return better

def capture(tile, rgb, better, reference):
    _capture(*inputs(tile['rgb'],tile['reference_rgb'],rgb,better),
             *geometry(tile,rgb.shape[1]),int(reference))
    _calls['native_capture'] += 1

def select(tile, covered, reference):
    counts = np.empty(2,np.int64)
    _select(*inputs(tile['rgb'],tile['reference_rgb'],tile['owner'],tile['labels'],
                    tile['best'],tile['reference_score'],tile['mask'],covered),
            *geometry(tile,covered.shape[1]),reference,ptr(counts))
    _calls['native_select'] += 1
    return int(counts[0]),int(counts[1])

def blend(tile, output, covered, distance, fade):
    _blend(*inputs(output,tile['rgb'],covered,distance),float(fade),*geometry(tile,output.shape[1]))
    _calls['native_blend'] += 1

class RenderAccumulator:
    def __init__(self,labels,band):
        self.shape = labels.shape
        labels = np.ascontiguousarray(labels,dtype=np.uint16)
        band = np.ascontiguousarray(band,dtype=np.uint8)
        self.handle = _create(ptr(labels),ptr(band),labels.size)
        if not self.handle:
            raise MemoryError('C++ render state allocation failed')
        self.seam_count = int(_seams(self.handle))
        _calls['render_create'] += 1

    def owner(self,index):
        mask = np.empty(self.shape,np.uint8)
        count = int(_owner(self.handle,index,ptr(mask)))
        return mask,count

    def add(self,index,rgb,valid,feather,output):
        _add(self.handle,index,ptr(rgb),ptr(valid),ptr(feather) if feather is not None else None,ptr(output))
        _calls['render_add'] += 1

    def finish(self,output):
        count = int(_finish(self.handle,ptr(output)))
        _calls['render_finish'] += 1
        return count

    def close(self):
        if getattr(self,'handle',None):
            _destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
