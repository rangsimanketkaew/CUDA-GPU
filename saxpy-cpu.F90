! gfortran -o saxpy-cpu saxpy-cpu.F90

module mathOps
    contains
       subroutine saxpy(x, y, a)
        implicit none
        real :: x(:), y(:), a
        ! Just a simple array scaler multiplication and addition
        y = a*x +y
      end subroutine saxpy 
    end module mathOps
    
    program testSaxpy
      use mathOps
      implicit none
      integer, parameter :: N = 40000
      real :: x(N), y(N), a

      x = 1.0; y = 2.0; a = 2.0

      write(*,*) 'Max error: ', maxval(abs(y-4.0))
    end program testSaxpy 
    