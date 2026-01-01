Target Definition

Purpose of the Target
The purpose of defining a target variable in this project is to clearly specify what is meant by a “bad loan” from a credit risk perspective.

Definition of Default
In this project, a loan is considered to have defaulted if it enters a state of charge-off within a fixed performance window after approval. A loan is labeled as a default if its loan status is recorded as “Charged Off” or “Default” during the observation period. Loans that are marked as “Fully Paid” are considered non-defaults. Loans with intermediate statuses such as “Late (16–30 days)” or “Late (31–120 days)” are not directly treated as defaults, as these states do not necessarily indicate permanent credit loss.

Performance Window
To ensure that each loan has sufficient time to demonstrate its repayment behavior, a fixed performance window is applied. Only loans that have had at least 12 months of performance history since origination are included in the target definition.
Loans that are too recent and have not yet completed the performance window are excluded from modeling, as their final outcomes are not fully observable.