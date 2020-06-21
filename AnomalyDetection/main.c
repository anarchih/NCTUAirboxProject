#include <Python.h>
#include "cython_wrapper.h"
#include <stdio.h> 


int init(){
  // Init module & Python 直譯器 
  PyImport_AppendInittab("cython_wrapper", PyInit_cython_wrapper);
  Py_Initialize();
  PyRun_SimpleString("import sys\nsys.path.insert(0,'')");
  PyImport_ImportModule("cython_wrapper");
}
int call(){
 
  // 呼叫 function
  wchar_t* str1=L"中科管理局";
  float values[2] = {20, 20};
  char signal[10];
  detect("2019/04/30 11:59:32", str1, values, signal);
  printf("Signal: %s\n", signal);
  //printf("%s\n", detect("123", str1, values));
  //printf("%s\n", detect("123", str1, values));
  //printf("%s\n", detect("123", str1, values));
  // send_arr(values, 3);
  // train("training.csv#training.csv");
  // printf("%d\n", detect("123", str1, 4));

}
int main(){
  init();
  call();
  Py_Finalize();
}
