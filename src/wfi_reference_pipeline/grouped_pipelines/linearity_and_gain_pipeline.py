"""
Placeholder for Tim Brandt analysis where the integral non-linearity, 
classic non-linearity and classic inverse non-linearity, and gain analysis
code should go.

If script has a file list for an input, and is assumed to be run per detector, 
and solves simultaneously for different reference file type arrays, such as:

lin_arr, invlin_arr = LinearityCorrections([filelist], kwrgs*)

then instances of reference file types Linearity and InverseLinearity can be instantied 
with the arrays as inputs into those respective classes like:

rfp_lin = Linearity(meta.linearity, ref_type_data=lin_arr)

rfp_invlin = InverseLinearity(meta.inverselinearity, ref_type_data=invlin_arr)
"""