"""C++ v3 pixel kernels; OpenCV supplies its existing native image operators."""
import ctypes as C
from collections import Counter
from pathlib import Path
from functools import wraps
import inspect
import cv2
import numpy as np
from .native_runtime import NativeLibrary, array, image, index as source_index
from . import fast_numpy

LIBRARY = Path(__file__).with_name('fast_core.dll')
_calls = Counter()
P, I, N, F, U = C.c_void_p, C.c_int, C.c_int64, C.c_float, C.c_uint16
_signatures = {
    'rgb_chroma': ([P,P,I,I,N], None), 'independent_detail': ([P,P,P,F,P,N], None),
    'focus_fields': ([P]*6+[F]+[P]*3+[N], None),
    'proxy_winners': ([P]*5+[U]+[P]*8+[N], None),
    'preserve_detail': ([P]*7+[N], None), 'neutral_seed_filter': ([P]*7+[N], None),
    'neutral_update': ([P]*6+[U,N], None), 'neutral_apply': ([P]*6+[N], None),
    'support_inputs': ([P]*4+[N], None), 'nearest_support': ([P]*5+[F,P,N], I),
    'contour_energy': ([P]*5+[N], None), 'normalize_field': ([P,P,N], None),
    'native_rank': ([P]*7+[I]*5+[U,I], None), 'native_capture': ([P]*4+[I]*6, None),
    'native_select': ([P]*8+[I]*5+[U,P], None), 'native_blend': ([P]*4+[F]+[I]*5, None),
    'render_create': ([P,P,N], P), 'render_destroy': ([P], None),
    'render_seams': ([P], N), 'render_owner': ([P,U,P], N),
    'render_add': ([P,U,P,P,P,P], None), 'render_finish': ([P,P], N),
}
_library = NativeLibrary(LIBRARY, _signatures, abi_name='fast_core_abi', expected_abi=3)
# Each new export is optional independently; the established ABI remains usable.
_hugin_libraries = {
    name: NativeLibrary(LIBRARY, {name: signature}, abi_name='fast_core_abi', expected_abi=3)
    for name, signature in {
        'structure_tensor_texture': ([P]*5+[N], None),
        'hard_mask': ([P,U,P,N], None),
    }.items()
}

def bind(name, args, result=None):
    return lambda *values: _library.call(name, *values)

_abi = lambda: _library.abi
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
    return {**_library.info(), 'calls': dict(_calls), 'calls_scope': 'process_lifetime',
            'hugin_kernels': {name: lib.info() for name, lib in _hugin_libraries.items()}}

def structure_tensor_texture(xx, yy, xy, energy):
    array(xx, np.float32)
    if xx.ndim != 2 or min(xx.shape) < 1:
        raise ValueError('nonempty tensor plane required')
    for value in (yy, xy, energy):
        array(value, np.float32, xx.shape)
    library = _hugin_libraries['structure_tensor_texture']
    _calls['structure_tensor_texture'] += 1
    if not library.available:
        return fast_numpy.structure_tensor_texture(xx, yy, xy, energy)
    out = np.empty(xx.shape, bool)
    library.call('structure_tensor_texture', *inputs(xx, yy, xy, energy, out), xx.size)
    return out

def hard_mask(labels, index):
    array(labels, np.uint16)
    if labels.ndim != 2 or min(labels.shape) < 1:
        raise ValueError('nonempty labels plane required')
    index = source_index(index)
    library = _hugin_libraries['hard_mask']
    _calls['hard_mask'] += 1
    if not library.available:
        return fast_numpy.hard_mask(labels, index)
    out = np.empty(labels.shape, np.uint8)
    library.call('hard_mask', ptr(labels), index, ptr(out), labels.size)
    return out

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
        self.handle = None
        self._state = None
        self._closed = True
        array(labels, np.uint16)
        if labels.ndim != 2 or min(labels.shape) < 1:
            raise ValueError('nonempty labels plane required')
        array(band, (np.bool_, np.uint8), labels.shape)
        self.shape = labels.shape
        band = np.ascontiguousarray(band,dtype=np.uint8)
        try:
            if _library.available:
                self.handle = _create(ptr(labels),ptr(band),labels.size)
                if not self.handle:
                    raise MemoryError('C++ render state allocation failed')
                self.seam_count = int(_seams(self.handle))
            else:
                self._state = fast_numpy.Render(labels, band)
                self.seam_count = self._state.seam.size
            self._closed = False
            _calls['render_create'] += 1
        except BaseException:
            self.close()
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError('render accumulator is closed')

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False

    def owner(self,index):
        self._check_open()
        index = source_index(index)
        if self._state is not None:
            return self._state.owner(index, self.shape)
        mask = np.empty(self.shape,np.uint8)
        count = int(_owner(self.handle,index,ptr(mask)))
        return mask,count

    def add(self,index,rgb,valid,feather,output):
        self._check_open()
        index = source_index(index)
        image(rgb); array(rgb, np.uint8, (*self.shape,3))
        array(valid, (np.bool_,np.uint8), self.shape)
        image(output, write=True); array(output, np.uint8, rgb.shape, write=True)
        if feather is not None:
            array(feather, np.float32, self.shape)
        if self._state is not None:
            self._state.add(index,rgb,valid,feather,output)
            _calls['render_add'] += 1
            return
        _add(self.handle,index,ptr(rgb),ptr(valid),ptr(feather) if feather is not None else None,ptr(output))
        _calls['render_add'] += 1

    def finish(self,output):
        self._check_open()
        array(output, np.uint8, (*self.shape,3), write=True)
        count = (self._state.finish(output) if self._state is not None else int(_finish(self.handle,ptr(output))))
        _calls['render_finish'] += 1
        return count

    def close(self):
        if getattr(self,'handle',None):
            handle = self.handle
            self.handle = None
            _destroy(handle)
        self._state = None
        self._closed = True

    def __del__(self):
        self.close()


_float_names = {'detail','broad','gradient','xx','yy','xy','variance','score','best',
                'detail_best','local_best','propagated_detail','strength','details','distance'}
_label_names = {'labels','propagated','local_owner','owner','detail_owner'}
_byte_names = {'chroma','rgb','output'}
_written = {
    'proxy_winners': {'best','labels','detail_best','detail_owner','local_best','local_owner','textured'},
    'neutral_update': {'best','owner','chroma'}, 'select': {'covered'}, 'blend': {'output'},
}


def _validate(name, values):
    if name == 'chroma':
        rgb = values['rgb']
        if not isinstance(rgb,np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError('RGB uint8 image required')
        values['rgb'] = np.ascontiguousarray(rgb)
        image(values['rgb'])
        return
    if 'index' in values:
        values['index'] = source_index(values['index'])
    if name == 'select':
        values['reference'] = source_index(values['reference'])
    for scalar in ('floor','radius','fade'):
        if scalar in values and (not np.isfinite(values[scalar]) or (scalar in ('radius','fade') and values[scalar] <= 0)):
            raise ValueError(f'invalid {scalar}')
    if name == 'nearest_support':
        values['strength'] = np.ascontiguousarray(values['strength'], dtype=np.float32)
        values['edge'] = np.ascontiguousarray(values['edge'] != 0)
    if name == 'rank':
        values['score'] = np.ascontiguousarray(values['score'], dtype=np.float32)
    if 'tile' in values:
        tile = values['tile']
        full = values[{'rank':'valid', 'capture':'rgb', 'select':'covered', 'blend':'output'}[name]]
        if name in ('capture','blend'):
            image(full, write=name == 'blend')
        else:
            array(full, (np.bool_,np.uint8), write=name == 'select')
            if full.ndim != 2:
                raise ValueError('full mask plane required')
        h,w = full.shape[:2]
        x,y,x1,y1 = (tile[key] for key in ('x','y','x1','y1'))
        if any(not isinstance(v,(int,np.integer)) for v in (x,y,x1,y1)) or not (0 <= x < x1 <= w and 0 <= y < y1 <= h):
            raise ValueError('tile outside full image')
        shape = (y1-y,x1-x)
        required = {
            'rank': ('best','owner','reference_score','mask'),
            'capture': ('rgb','reference_rgb'),
            'select': ('rgb','reference_rgb','owner','labels','best','reference_score','mask'),
            'blend': ('rgb',),
        }[name]
        writable = {'rank': {'best','owner','reference_score'},'capture': {'rgb','reference_rgb'},
                    'select': {'rgb','owner'},'blend': set()}[name]
        for key in required:
            dtype = np.float32 if key in ('best','reference_score') else np.uint16 if key in ('owner','labels') else np.uint8 if key in ('rgb','reference_rgb') else (np.bool_,np.uint8)
            array(tile[key],dtype,(*shape,3) if key in ('rgb','reference_rgb') else shape,write=key in writable,name=key)
        if name == 'rank': array(values['score'],np.float32,shape)
        if name == 'capture': array(values['better'],(np.bool_,np.uint8),shape)
        if name == 'blend':
            array(values['covered'],(np.bool_,np.uint8),(h,w))
            array(values['distance'],np.float32,(h,w))
        return
    arrays = {key:value for key,value in values.items() if isinstance(value,np.ndarray)}
    anchor = arrays.get('local_best') if name == 'neutral_filter' else next(iter(arrays.values()))
    if anchor.ndim != 2 or min(anchor.shape) < 1:
        raise ValueError('nonempty pixel plane required')
    shape = anchor.shape
    for key,value in arrays.items():
        expected = values['positions'].shape if name == 'neutral_filter' and key in ('positions','details') else shape
        dtype = np.int64 if key == 'positions' else np.float32 if key in _float_names else np.uint16 if key in _label_names else np.uint8 if key in _byte_names else (np.bool_,np.uint8)
        array(value,dtype,expected,write=key in _written.get(name,set()),name=key)
    if name == 'proxy_winners' and values['textured'].dtype == np.bool_:
        # A Boolean output needs canonical bytes even when an input mask is 255.
        values['texture'] = np.ascontiguousarray(values['texture'], dtype=np.bool_)
    if name == 'neutral_filter':
        positions = values['positions']
        if positions.size and (positions.min() < 0 or positions.max() >= anchor.size):
            raise ValueError('neutral seed position outside image')


def _dispatch(fn):
    signature = inspect.signature(fn)
    @wraps(fn)
    def wrapped(*args, **kwargs):
        values = signature.bind(*args, **kwargs)
        _validate(fn.__name__, values.arguments)
        if _library.available:
            return fn(*values.args, **values.kwargs)
        key = {'independent':'independent_detail', 'neutral_filter':'neutral_seed_filter',
               'rank':'native_rank', 'capture':'native_capture', 'select':'native_select',
               'blend':'native_blend'}.get(fn.__name__, fn.__name__)
        _calls[key] += 1
        return getattr(fast_numpy, fn.__name__)(*values.args, **values.kwargs)
    return wrapped


for _name in ('chroma','independent','focus_fields','proxy_winners','preserve_detail','nearest_support',
              'neutral_filter','neutral_update','neutral_apply','contour_energy','rank','capture','select','blend'):
    globals()[_name] = _dispatch(globals()[_name])
