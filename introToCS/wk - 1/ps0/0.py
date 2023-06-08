#Write a program that does the following in order:
# 1. Asks the user to enter a number “x”
# 2. Asks the user to enter a number “y” 
# 3. Prints out number “x”, raised to the power “y”.
# 4. Prints out the log (base 2) of “x”

import math

user_x_input = input('Enter number x:')
num_x = int(user_x_input)

user_y_input = input('Enter number y:')
num_y = int(user_y_input)

print('X**Y =', num_x ** num_y)
print('log(x) = ', math.log(num_x))
