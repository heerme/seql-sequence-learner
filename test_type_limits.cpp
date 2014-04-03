#include <iostream>
#include <limits>
#include <typeinfo>
#include <iomanip>
using namespace std ;

template<typename T> void show_limits()
{
  typedef numeric_limits<T> limits ;
  cout << "********************* type " << typeid(T).name() 
         << " **************************\n" ;
  cout << "size: "
  	   	<< sizeof(T) << "\n";
  cout << "smallest nonzero denormalized value: " 
         << limits::denorm_min() << '\n' ;
  cout << "allows  denormalized values? " << boolalpha 
         << (limits::has_denorm==denorm_present) << '\n' ;
  cout << "difference between 1 and the smallest value greater than 1: "  
         << limits::epsilon() << '\n' ;
  cout << "maximum rounding error: " << limits::round_error() << '\n' ;
  cout << "base (radix) used for the representation: " 
         << limits::radix << '\n' ;
  cout << "minimum value of exponent (radix): " 
         << limits::min_exponent << '\n' ;
  cout << "approximate minimum value of exponent (decimal): " 
         << limits::min_exponent10 << '\n' ;
  cout << "maximum value of exponent (radix): " 
         << limits::max_exponent << '\n' ;
  cout << "approximate maximum value of exponent (decimal): " 
         << limits::max_exponent10 << '\n' ;
  cout << "minimum normalized value: " << limits::min() << '\n' ;
  cout << "maximum normalized value: " << limits::max() << "\n\n" ;
 
}

int main()
{
  show_limits<unsigned int>() ;
  show_limits<int>() ;
  show_limits<float>() ;
  show_limits<double>() ;
  show_limits<long double>() ;

  double  x = 1.0e+20, y = 1.0e-20, z = -1.0e+20 ;
  cout << scientific << "x+y+z: " << x+y+z << '\n' ;
  cout << "x+z+y: " << x+z+y << '\n' ;
  cout << fixed << setprecision(50)  << "x+y+z: " << x+y+z << '\n' ;
  cout << "x+z+y: " << x+z+y << '\n' ;
}
