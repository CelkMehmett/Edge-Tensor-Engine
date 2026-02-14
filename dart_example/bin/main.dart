import 'package:edge_tensor_engine_dart/tensor_engine.dart';
import 'package:path/path.dart' as path;
import 'dart:io';

void main() {
  // Assume library is in target/debug/ or target/release/
  // Adjust path relative to current script
  final libName = Platform.isLinux ? 'libedge_tensor_engine.so' : 
                  Platform.isMacOS ? 'libedge_tensor_engine.dylib' : 
                  'edge_tensor_engine.dll';
  
  // This path assumes running from root of repo or dart_example root
  // Try several locations
  var libPath = path.join(Directory.current.path, '..', 'target', 'debug', libName);
  if (!File(libPath).existsSync()) {
      libPath = path.join(Directory.current.path, 'target', 'debug', libName);
  }
  if (!File(libPath).existsSync()) {
      libPath = path.join(Directory.current.path, '..', 'target', 'release', libName);
  }
  if (!File(libPath).existsSync()) {
      libPath = path.join(Directory.current.path, 'target', 'release', libName);
  }
  
  print("Loading library from: $libPath");
  
  final engine = TensorEngine(libPath);
  
  print("Creating Tensors...");
  // A: [2, 3]
  final a = engine.createTensor(
      [1.0, 2.0, 3.0, 
       4.0, 5.0, 6.0], 
      [2, 3]
  );
  
  // B: [3, 2]
  final b = engine.createTensor(
      [1.0, 0.0, 
       0.0, 1.0, 
       1.0, 1.0], 
      [3, 2]
  );
  
  print("Performing Matmul A @ B...");
  final c = engine.matmul(a, b);
  
  final result = engine.getData(c);
  print("Result C (flattened): $result");
  // Expected:
  // [1*1+2*0+3*1, 1*0+2*1+3*1] = [1+3, 2+3] = [4, 5]
  // [4*1+5*0+6*1, 4*0+5*1+6*1] = [4+6, 5+6] = [10, 11]
  // Result: [4.0, 5.0, 10.0, 11.0]

  // Verify
  if (result.length == 4 && 
      result[0] == 4.0 && result[1] == 5.0 && 
      result[2] == 10.0 && result[3] == 11.0) {
      print("SUCCESS: Matmul correct.");
  } else {
      print("FAILURE: Matmul incorrect.");
  }
  
  print("Cleaning up...");
  engine.freeTensor(a);
  engine.freeTensor(b);
  engine.freeTensor(c);
  print("Done.");
}
