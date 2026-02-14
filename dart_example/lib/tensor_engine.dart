import 'dart:ffi' as ffi;
import 'dart:io' show Platform, Directory;
import 'package:path/path.dart' as path;
import 'package:ffi/ffi.dart';

// --- FFI Type Definitions ---

typedef TensorHandle = ffi.Pointer<ffi.Void>;

// tensor_create_f32(data: *const f32, shape_ptr: *const i64, ndim: usize) -> *mut Tensor
typedef TensorCreateF32C = TensorHandle Function(
    ffi.Pointer<ffi.Float> data,
    ffi.Pointer<ffi.Int64> shape,
    ffi.IntPtr ndim);
typedef TensorCreateF32Dart = TensorHandle Function(
    ffi.Pointer<ffi.Float> data,
    ffi.Pointer<ffi.Int64> shape,
    int ndim);

// tensor_free(ptr: *mut Tensor)
typedef TensorFreeC = ffi.Void Function(TensorHandle tensor);
typedef TensorFreeDart = void Function(TensorHandle tensor);

// tensor_matmul(lhs, rhs) -> *mut Tensor
typedef TensorMatmulC = TensorHandle Function(TensorHandle lhs, TensorHandle rhs);
typedef TensorMatmulDart = TensorHandle Function(TensorHandle lhs, TensorHandle rhs);

// tensor_data_ptr(tensor) -> *const f32
typedef TensorDataPtrC = ffi.Pointer<ffi.Float> Function(TensorHandle tensor);
typedef TensorDataPtrDart = ffi.Pointer<ffi.Float> Function(TensorHandle tensor);

// tensor_get_shape(tensor, out_ndim)
typedef TensorGetShapeC = ffi.Pointer<ffi.IntPtr> Function(TensorHandle tensor, ffi.Pointer<ffi.IntPtr> outNdim);
typedef TensorGetShapeDart = ffi.Pointer<ffi.IntPtr> Function(TensorHandle tensor, ffi.Pointer<ffi.IntPtr> outNdim);

// tensor_backward(root)
typedef TensorBackwardC = ffi.Void Function(TensorHandle root);
typedef TensorBackwardDart = void Function(TensorHandle root);

// tensor_grad(tensor) -> *mut Tensor
typedef TensorGradC = TensorHandle Function(TensorHandle tensor);
typedef TensorGradDart = TensorHandle Function(TensorHandle tensor);

// --- Wrapper Class ---

class TensorEngine {
  late ffi.DynamicLibrary _dylib;
  
  late TensorCreateF32Dart _tensorCreateF32;
  late TensorFreeDart _tensorFree;
  late TensorMatmulDart _tensorMatmul;
  late TensorDataPtrDart _tensorDataPtr;
  late TensorGetShapeDart _tensorGetShape;
  late TensorBackwardDart _tensorBackward;
  late TensorGradDart _tensorGrad;

  TensorEngine(String libPath) {
    _dylib = ffi.DynamicLibrary.open(libPath);
    
    _tensorCreateF32 = _dylib
        .lookup<ffi.NativeFunction<TensorCreateF32C>>('tensor_create_f32')
        .asFunction();
        
    _tensorFree = _dylib
        .lookup<ffi.NativeFunction<TensorFreeC>>('tensor_free')
        .asFunction();
        
    _tensorMatmul = _dylib
        .lookup<ffi.NativeFunction<TensorMatmulC>>('tensor_matmul')
        .asFunction();
        
    _tensorDataPtr = _dylib
        .lookup<ffi.NativeFunction<TensorDataPtrC>>('tensor_data_ptr')
        .asFunction();
        
    _tensorGetShape = _dylib
        .lookup<ffi.NativeFunction<TensorGetShapeC>>('tensor_get_shape')
        .asFunction();

    try {
        _tensorBackward = _dylib
            .lookup<ffi.NativeFunction<TensorBackwardC>>('tensor_backward')
            .asFunction();
        _tensorGrad = _dylib
            .lookup<ffi.NativeFunction<TensorGradC>>('tensor_grad')
            .asFunction();
    } catch (e) {
        print("Warning: Autograd symbols not found (maybe not exported yet): $e");
    }
  }

  TensorHandle createTensor(List<double> data, List<int> shape) {
    final dataPtr = calloc<ffi.Float>(data.length);
    for (var i = 0; i < data.length; i++) {
      dataPtr[i] = data[i];
    }
    
    final shapePtr = calloc<ffi.Int64>(shape.length);
    for (var i = 0; i < shape.length; i++) {
        shapePtr[i] = shape[i];
    }
    
    final tensor = _tensorCreateF32(dataPtr, shapePtr, shape.length);
    
    calloc.free(dataPtr);
    calloc.free(shapePtr);
    
    return tensor;
  }

  void freeTensor(TensorHandle tensor) {
    if (tensor != ffi.nullptr) {
       _tensorFree(tensor);
    }
  }
  
  TensorHandle matmul(TensorHandle lhs, TensorHandle rhs) {
    return _tensorMatmul(lhs, rhs);
  }
  
  List<double> getData(TensorHandle tensor) {
      if (tensor == ffi.nullptr) return [];
      
      // Get shape to know size
      final ndimPtr = calloc<ffi.IntPtr>();
      final shapePtr = _tensorGetShape(tensor, ndimPtr);
      final ndim = ndimPtr.value;
      
      int numel = 1;
      // Note: shapePtr points to internal Vec<usize> data which is platform dependent size.
      // But we defined `get_shape` to return `*const usize`.
      // `ffi.IntPtr` matches `usize`.
      
      // Need to read array from pointer
      // shapePtr is generic Pointer, cast to specific is hard if array size unknown at compile time.
      // Can iterate pointer.
      
      // However, `usize` size varies (4 or 8 bytes). `IntPtr` handles this.
      for (int i = 0; i < ndim; i++) {
          numel *= shapePtr.elementAt(i).value;
      }
      
      calloc.free(ndimPtr);
      
      final dataPtr = _tensorDataPtr(tensor);
      final result = <double>[];
      for (int i = 0; i < numel; i++) {
          result.add(dataPtr[i]);
      }
      return result;
  }
  
  void backward(TensorHandle root) {
      _tensorBackward(root);
  }
  
  TensorHandle grad(TensorHandle tensor) {
      return _tensorGrad(tensor);
  }
}
