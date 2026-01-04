Target Variable Definition

The goal of this project is to predict whether a loan will default or be fully paid.
The target variable is created using the loan status provided in the dataset.

We define the target as a binary variable:

1 - Loan is Charged Off (customer failed to repay the loan)

0 - Loan is Fully Paid

Only loans with a final and known outcome are used for training the model.

Handling Ongoing Loans

Loans with status such as “Current” are excluded from the dataset.
These loans are still active, and their final outcome (default or full payment) is not yet known.

Including them would introduce uncertainty and make the model unrealistic, since at the time of prediction we only want to learn from loans with completed outcomes.

