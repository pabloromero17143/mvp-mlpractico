import matplotlib.pyplot as plt
from random import sample
from sklearn.model_selection import train_test_split

## Validamos si lo que sacamos nos impactó en el budget
def plot_budget_validation(df_filter, df_validation, log_var=True):
    (
        df_filter
            .budget
            .value_counts()
            .value_counts(normalize=True)
            .sort_index()
            .plot(logx=log_var, logy=log_var, label='movies_gross_filter')
    )
    (
        df_validation
            .budget
            .value_counts()
            .value_counts(normalize=True)
            .sort_index()
            .plot(logx=log_var, logy=log_var, label='movies_gross')
    )

    plt.legend(loc='best')
    plt.title('Budget of films per movie')

def load_budget_train_dev_test(df):

    X = df.loc[ : , df.columns != 'budget'] #me quedo con train solo
    y = df.budget.values  #var target y

    # Splite en 80:20, esto es train y val
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=101)

    # AHora con el 0.875, tengo mi test, y finalmente me queda 70,20,10   (ver que 0.8 * 0.875 = 0.7)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.875, random_state=101)

    print("len(X): {} len(y): {} \nlen(X_train): {}, len(X_val): {}, len(X_test): \
    {} \nlen(y_train): {}, len(y_val): {}, len(y_test): {}".format(len(X), len(y),\
    len(X_train), len(X_val), len(X_test), len(y_train), len(y_val), \
    len(y_test))) 
    
    return X_train.to_dict(orient='records'), X_val.to_dict(orient='records'), X_test.to_dict(orient='records'), y_train, y_val, y_test    
