# toal_cost = cost of dream home
# portion_down_payment = amt needed for down pmt
# current_savings = current savings
# r = apr
# end of the month recieve current_savings*r/12
# r = 0.04
# annual_salary = annual salary 
# portion_saved = amount saved every month from 
#compounds monthly 

#3 inputs:
#1. starting annual salary(annual_salary)
#2. portion of salary to be saved(portion_saved)
#3. cost of dream home(total_cost)

#how many months will it take to save upu enough money for a down payment?

#scenario 1:
class SavingsCalculator:
   def dreamHome(self):
    input_salary = input('Enter starting annual salary:')
    input_saved = input('Enter portion to be saved as decimal:')
    input_cost = input('Enter cost of dream home: ')

    annual_salary = float(input_salary)
    portion_saved = float(input_saved)
    total_cost = float(input_cost)

    monthly_salary = annual_salary/12

    r = 0.04
    monthly_r = r/12


    current_savings = 0
    months = 0
    savings_needed = total_cost*0.25
    

    while current_savings < savings_needed:
        current_savings += monthly_salary*portion_saved
        current_savings += current_savings*monthly_r
        months += 1

    print('Number of Months: ', months) 
        


savings_calculator = SavingsCalculator()
savings_calculator.dreamHome()

