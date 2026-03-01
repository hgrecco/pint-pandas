import pint_pandas
import pandas as pd
import pint

PA_ = pint_pandas.PintArray
Q_ = pint.Quantity

a = PA_([1, 2, 3], "m")
print(a* 1.2)