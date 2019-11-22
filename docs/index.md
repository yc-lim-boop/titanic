# Titanic

### Dataset

Task: predict survival of Titanic passengers

Size: 891 in train set, 418 in test set

Given features:
- Ticket class
- Name
- Sex
- Age
- \# of siblings/spouses on board
- \# of parents/children on board
- Ticket number
- Fare
- Cabin
- Embarkation location

## Data Exploration


```python
from pprint import pprint
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
sns.set(style='whitegrid')
```


```python
import altair as alt
alt.renderers.enable('notebook')
```




    RendererRegistry.enable('notebook')




```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
targets = train_data['Survived']
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data['Died'] = 1 - train_data['Survived']
```

### Missing data


```python
train_data.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    Died             0
    dtype: int64




```python
test_data.isna().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



### Univariate exploration


```python
p = sns.PairGrid(data=train_data,
                 y_vars='Survived',
                 x_vars=['Pclass', 'Sex', 'Embarked'],
                 height=3)
p.set(ylim=(0, 1))
p.map(sns.barplot, ci=None)
for ax in p.axes.flatten():
    ax.axhline(y=0.5, ls=':', color='k', lw=1)
sns.despine(fig=p.fig, left=True)
```


![png](output_12_0.png)



```python
sns.barplot(data=train_data, x='Pclass', y='Survived', hue='Sex', ci=None)
plt.ylim(0, 1)
```




    (0, 1)




![png](output_13_1.png)



```python
train_data['isChild'] = train_data['Age'].map(lambda age: age < 16)
sns.barplot(data=train_data, x='Sex', hue='isChild', ci=None, y='Survived')
plt.ylim(0, 1)
```




    (0, 1)




![png](output_14_1.png)



```python
p = sns.FacetGrid(data=train_data, row='Sex', col='Pclass', col_order=[1, 2, 3], margin_titles=True)
p.map(sns.barplot, 'isChild', 'Survived', ci=None)
for ax in p.axes.flatten():
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, ls=':', lw=1, color='k')
```

    C:\Users\lim.yaoc\AppData\Local\Continuum\anaconda3\lib\site-packages\seaborn\axisgrid.py:715: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)
    


![png](output_15_1.png)



```python
print(train_data.groupby(['Sex', 'Pclass', 'isChild'])['PassengerId'].count())
```

    Sex     Pclass  isChild
    female  1       False       91
                    True         3
            2       False       66
                    True        10
            3       False      114
                    True        30
    male    1       False      119
                    True         3
            2       False       99
                    True         9
            3       False      319
                    True        28
    Name: PassengerId, dtype: int64
    

#### Fare

Low fares have a lower chance of survival


```python
alt.Chart(train_data).mark_bar().encode(
    alt.X('Fare', bin=alt.Bin(step=10)),
    y='mean(Survived)'
).properties(width=800)
```


    <vega.vegalite.VegaLite at 0x270e867b2c8>





    




![png](output_19_2.png)



```python
alt.Chart(train_data).mark_bar().encode(
    alt.X('Fare', bin=alt.Bin(step=5)),
    y='mean(Survived)'
).properties(
    width=800
).facet(
    row='Sex:N'
)
```


    <vega.vegalite.VegaLite at 0x270e8673188>





    




![png](output_20_2.png)


#### Age

Lower age tends to mean higher chance of survival

Male + female


```python
base = alt.Chart(train_data)

surv = base.mark_bar().encode(
    alt.X('Age', bin=alt.Bin(step=1)),
    y='mean(Survived)'
)
counts = base.mark_bar().encode(
    alt.X('Age', bin=alt.Bin(step=1)),
    y='count()'
)
rule = alt.Chart(pd.DataFrame({'x': [16]})).mark_rule().encode(
    x='x:Q'
)

alt.vconcat((surv + rule).properties(width=800), (counts + rule).properties(width=800))
```


    <vega.vegalite.VegaLite at 0x270e86b7e08>





    




![png](output_24_2.png)


Male only (from 14-16, rate of survival drops)


```python
base = alt.Chart(train_data[train_data['Sex'] == 'male'])

surv = base.mark_bar().encode(
    alt.X('Age', bin=alt.Bin(step=1)),
    y='mean(Survived)'
)

rule = alt.Chart(pd.DataFrame({'x': [16]})).mark_rule().encode(
    x='x:Q'
)

counts = base.mark_bar().encode(
    alt.X('Age', bin=alt.Bin(step=1)),
    y='count()'
)

alt.vconcat((surv + rule).properties(width=800), (counts + rule).properties(width=800))
```


    <vega.vegalite.VegaLite at 0x270e87ebe08>



![png](output_26_1.png)





    




```python
alt.Chart(train_data).mark_bar().encode(
    alt.X('Age', bin=alt.Bin(step=1)),
    y='mean(Survived)'
).properties(width=800)
```


    <vega.vegalite.VegaLite at 0x270e83e6108>





    




![png](output_27_2.png)



```python
alt.Chart(train_data).mark_bar().encode(
    alt.X('Age', bin=alt.Bin(step=1)),
    y='mean(Survived)'
).properties(
    width=800
).facet(
    row='Sex:N'
)
```


    <vega.vegalite.VegaLite at 0x270e86b3c88>





    




![png](output_28_2.png)


#### Title

The title of each passenger (Mr., Mrs., Doctor. etc.)


```python
def get_title(name):
    return name.split(',')[1].split('.')[0].strip()

def get_category(title):
    return title_mapping.get(title, 'Other')

title_mapping = {
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Ms': 'Miss',
    'Mlle': 'Miss',
    'Master': 'Master',
    'Mme': 'Mrs',
    'Major': 'Officer',
    'Col': 'Officer',
    'Capt': 'Officer',
    'Don': 'Royalty',
    'Lady': 'Royalty',
    'Sir': 'Royalty',
    'the Countess': 'Royalty',
    'Jonkheer': 'Royalty',
    'Rev': 'Other',
    'Dr': 'Other',
    #'Rev': 'Officer',
    #'Dr': 'Officer',
    #'Dona': 'Royalty',  # only appears in test set
}
```


```python
train_data['Category'] = train_data['Name'].map(lambda n: get_category(get_title(n)))
```


```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15, 6))
sns.barplot('Category', 'Survived', data=train_data, ci=None, ax=ax1)
ax1.axhline(y=0.5, color='k', ls=':', lw=2)
ax1.set_ylim(0, 1)
_ = sns.countplot('Category', data=train_data, ax=ax2)
```


![png](output_33_0.png)


#### Cabins and Decks

Handling missing data for cabin: We just put 'unknown'

The cabins have the format &lt;Letter&gt; &lt;Number&gt;. The letter indicates which deck the cabin is on (A is the highest deck, G is the lowest), so this might be a predictive feature for survival.


```python
train_data['Cabin'].fillna('U', inplace=True)
train_data['Deck'] = train_data['Cabin'].map(lambda c: c[0])
```


```python
for deck in sorted(train_data['Deck'].unique()):
    subset = train_data[train_data['Deck'] == deck]
    frac = subset.sum()['Survived'] / len(subset)
    print(f'{deck}\t{len(subset)}\t{frac:.2f}')
```

    A	15	0.47
    B	47	0.74
    C	59	0.59
    D	33	0.76
    E	32	0.75
    F	13	0.62
    G	4	0.50
    T	1	0.00
    U	687	0.30
    


```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15, 6))
sns.barplot('Deck', 'Survived', data=train_data, ci=None, order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], ax=ax1)
ax1.set_ylim(0, 1)
ax1.axhline(0.5, ls=':', color='k')
ax2.set_ylim(0, 70)
_ = sns.countplot('Deck', data=train_data, ax=ax2)
```


![png](output_38_0.png)


#### Family size


```python
def family_size(row):
    return row['SibSp'] + row['Parch']
```


```python
train_data['FamilySize'] = train_data.apply(family_size, axis=1)
```


```python
p = sns.PairGrid(data=train_data,
                 y_vars='Survived',
                 x_vars=['FamilySize', 'SibSp', 'Parch'],
                 height=3)
p.set(ylim=(0, 1))
p.map(sns.pointplot, scale=1)
for ax in p.axes.flatten():
    ax.axhline(y=0.5, ls=':', lw=1, color='k')
sns.despine(fig=p.fig, left=True)
```


![png](output_42_0.png)


## Features

Features implemented:
- Imputation of missing values
- Title + Category
- Family features
- Ticket features
- Cabin features
- Combination features

### Title + Category


```python
def get_title(name):
    return name.split(',')[1].split('.')[0].strip()

def get_category(title):
    return title_mapping.get(title, 'Other')

title_mapping = {
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Ms': 'Miss',
    'Mlle': 'Miss',
    'Master': 'Master',
    'Mme': 'Mrs',
    'Major': 'Officer',
    'Col': 'Officer',
    'Capt': 'Officer',
    'Don': 'Royalty',
    'Lady': 'Royalty',
    'Sir': 'Royalty',
    'the Countess': 'Royalty',
    'Jonkheer': 'Royalty',
    'Rev': 'Other',
    'Dr': 'Other',
    #'Rev': 'Officer',
    #'Dr': 'Officer',
    #'Dona': 'Royalty',  # only appears in test set
}
```

### Family features
- size of family
- Categorize into big, small, and non-families (since families > 4 seem to have higher rates of death)
- whether someone from the family survived

Approach for last feature:
1. From training set, get list of last names. Assume those with the same last name are from the same family.
2. Names that have no family (or SibSp + Parch == 0) are set to 'U'. 
3. Names with family: those with family who survived (in the training set) get 'Y', 
   those without get 'N'.


```python
def have_family(row):
    if row['SibSp'] > 0 or row['Parch'] > 0:
        return 1
    else:
        return 0

def family_size(row):
    return row['SibSp'] + row['Parch']

def family_size_cat(row):
    fam_size = family_size(row)
    if fam_size == 0:
        return 'Single'
    elif fam_size < 4:
        return 'Small'
    else:
        return 'Big'
    
def last_name(name):
    return name.split(',')[0].strip()

def get_survived_names(data):
    return set(data.loc[data['Survived'] == 1]['Name'].map(last_name))

def get_family_status(data):
    survived_names = get_survived_names(data)
    all_names = set(data['Name'].apply(last_name))
    status = {
        n: True for n in survived_names
    }
    status.update({
        n: False for n in (all_names - survived_names)
    })
    return status

def get_family_survived(row, family_status):
    l_name = last_name(row['Name'])
    
    if (row['SibSp'] == 0 and row['Parch'] == 0) or l_name not in family_status:
        return 'U'
    elif family_status[l_name]:
        return 'Y'
    else:
        return 'N'

# also join family name with family size??

def add_family_features(data):
    data['FamilySize'] = data.apply(family_size, axis=1)
    data['HaveFamily'] = data.apply(have_family, axis=1)
    data['FamilyCategory'] = data.apply(family_size_cat, axis=1)
    
    train_data, _ = split_data(data)

    family_status = get_family_status(train_data)
    data['FamilySurvived'] = data.apply(lambda row: get_family_survived(row, family_status), axis=1)

    return data
```

### Ticket features

- \# of digits in the ticket number
- The first letter of the ticket (or whether the ticket only had numbers)
- The prefix of the ticket (i.e. the part before the number)
- Whether someone on the ticket survived (different from family, since some families had nannies, or had unrelated people on the same ticket)
  - Only have this feature for tickets with >1 person. Otherwise put 'U' (unknown)


```python
prefixes = train_data['Ticket'].map(lambda tick: tick.rsplit(maxsplit=1)[0] if not tick.isdigit() else 'DIGIT')
pprint(prefixes.value_counts())
```

    DIGIT          661
    PC              60
    C.A.            27
    STON/O 2.       12
    A/5             10
    W./C.            9
    CA.              8
    SOTON/O.Q.       8
    SOTON/OQ         7
    A/5.             7
    STON/O2.         6
    CA               6
    C                5
    SC/PARIS         5
    S.O.C.           5
    F.C.C.           5
    LINE             4
    SC/Paris         4
    A/4.             3
    PP               3
    S.O./P.P.        3
    A/4              3
    SC/AH            2
    S.C./PARIS       2
    A./5.            2
    P/PP             2
    A.5.             2
    WE/P             2
    SOTON/O2         2
    S.C./A.4.        1
    C.A./SOTON       1
    SCO/W            1
    F.C.             1
    A/S              1
    A4.              1
    SC               1
    SW/PP            1
    SC/AH Basle      1
    SO/C             1
    S.W./PP          1
    Fa               1
    S.O.P.           1
    S.P.             1
    W.E.P.           1
    W/C              1
    Name: Ticket, dtype: int64
    


```python
"""
Ticket:

- Number of digits in number
- whether number only
- first letter (if not number only)
- number part
"""

def n_digits(ticket):
    return len(ticket.split()[-1])

def first_letter(ticket):
    if ticket.isdigit():
        return 'DIGIT'
    else:
        return ticket[0]

def number_only(ticket):
    return ticket.split()[-1]

def ticket_prefix(ticket):
    """
    do some normalization, since some tickets are identical except for punctuation
    
    e.g.
    - A/5., A./5., A.5.
    - A/4, A4., A/4.
    - SOTON/OQ, SOTON/O.Q.
    """
    ticket = ticket.replace('.', '').replace('/', '').strip().split()
    if ticket[0].strip().isdigit():
        return 'DIGIT'
    else:
        return ticket[0].strip().upper()
    
"""
Consolidate rare ticket prefixes
"""

def make_prefix_mapping(prefixes):
    threshold = 5
    prefix_counts = Counter(prefixes)
    return {
        tp: tp if prefix_counts[tp] >= threshold else 'Other'
        for tp in prefix_counts.keys()
    }

def add_ticket_prefixes(data):
    train_data, _ = split_data(data)
    train_prefixes = train_data['Ticket'].map(ticket_prefix)
    
    tp_mapping = make_prefix_mapping(train_prefixes)
    prefixes = data['Ticket'].map(ticket_prefix)
    data['TicketPrefix'] = prefixes.map(lambda tp: tp_mapping.get(tp, 'Other'))
    
    return data
    
def add_ticket_features(data):
    tickets = data['Ticket']
    data['TicketNDigits'] = tickets.map(n_digits)
    data['TicketLetter'] = tickets.map(first_letter)
    #data['TicketNumber'] = tickets.map(number_only)
    data = add_ticket_prefixes(data)
    data['TicketSurvived'] = add_ticket_survived(data)
    return data
```


```python
"""
Ticket survived

Some cabins had nannies/unrelated people on the same ticket.
So also have a feature indicating whether someone on the same Ticket survived

Only have this feature for tickets with > 1 person. Otherwise put 'U' (unknown)
"""

def survivors(data):
    return data.loc[data['Survived'] == 1]

def get_survived_tickets(data):
    return set(survivors(data)['Ticket'])

def get_ticket_status(data):
    ticket_counts = Counter(data['Ticket'])
    non_single_tickets = set(k for (k, v) in ticket_counts.items() if v > 1)
    
    survived_tickets = get_survived_tickets(data)
    
    status = {
        t: True for t in (non_single_tickets & survived_tickets)
    }
    status.update({
        t: False for t in (non_single_tickets - survived_tickets)
    })
    
    return status

def get_ticket_survived(ticket, ticket_status):
    if ticket not in ticket_status:
        return 'U'
    elif ticket_status[ticket]:
        return 'Y'
    else:
        return 'N'

def add_ticket_survived(data):
    train_data, _ = split_data(data)
    ticket_status = get_ticket_status(train_data)
    return data.apply(lambda row: get_ticket_survived(row['Ticket'], ticket_status), axis=1)
```

### Cabin features

Some cabins had nannies/unrelated people on the same ticket.
So also have a feature indicating whether someone in the same Cabin survived

Only have this feature for cabins with >1 person. Otherwise put 'U' (unknown)


```python
train_data['Cabin'].value_counts()
```




    U              687
    B96 B98          4
    C23 C25 C27      4
    G6               4
    F33              3
                  ... 
    T                1
    D28              1
    B30              1
    D56              1
    E10              1
    Name: Cabin, Length: 148, dtype: int64




```python
def get_survived_cabins(data):
    return set(survivors(data)['Cabin'])

def get_cabin_status(data):
    cabin_counts = Counter(data['Cabin'])
    non_single_cabins = set(k for (k, v) in cabin_counts.items() if v > 1)
    non_single_cabins.discard('U')
    
    survived_cabins = get_survived_cabins(data)
    
    status = {
        t: True for t in (non_single_cabins & survived_cabins)
    }
    status.update({
        t: False for t in (non_single_cabins - survived_cabins)
    })
    
    return status

def get_cabin_survived(cabin, cabin_status):
    if cabin not in cabin_status:
        return 'U'
    elif cabin_status[cabin]:
        return 'Y'
    else:
        return 'N'

def add_cabin_survived(data):
    train_data, _ = split_data(data)
    cabin_status = get_cabin_status(train_data)
    return data.apply(lambda row: get_cabin_survived(row['Cabin'], cabin_status), axis=1)
```

### Misc features

- Whether their age is >= 16 (adult)
- Combination of Sex and Pclass
- Whether they are a male adult


```python
def sex_and_class(row):
    s = row['Sex']
    c = row['Pclass']

    return f"{s}&{c}"

child_max = 16

def is_child(age):
    if age < child_max:
        return 1
    else:
        return 0

def is_male_adult(row):
    if row['Sex'] == 'male' and row['Age'] >= child_max:
        return 1
    else:
        return 0
```

### Imputation

Features to impute:
- Age (177 in train, 86 in test)
- Fare (1 in test)
- Cabin (687 in train, 327 in test)
- Embarked (2 in train)

#### Age imputation

Fill with the median of the same Sex, Pclass, Category


```python
def impute_age(data):
    indices = ['Sex', 'Pclass', 'Category']

    train_data, _ = split_data(data)

    median_ages = train_data.dropna(subset=['Age']).groupby(indices).median()['Age']
    median_ages = dict(zip(median_ages.index.to_flat_index(), median_ages.to_list()))
        
    def find_median(row):
        idx = tuple(row[k] for k in indices)
        return median_ages[idx]
    
    data['Age'] = data.apply(
        lambda row: find_median(row)
        if np.isnan(row['Age']) else row['Age'],
        axis=1
    )

    return data
```


```python
def impute_age_old(data):
    indices = ['Sex', 'Pclass', 'Category']

    train_data, _ = split_data(data)

    median_ages = train_data.dropna(subset=['Age']).groupby(indices).median()['Age']
    median_ages = dict(zip(median_ages.index.to_flat_index(), median_ages.to_list()))
    
    median_ages = {None: train_data['Age'].median()}
    for i in range(len(indices)):
        prefix = indices[:i+1]
        ages = train_data.dropna(subset=['Age']).groupby(prefix).median()['Age']
        median_ages.update(dict(zip(ages.index.to_flat_index(), ages.to_list())))

        
    def find_median(row):
        idx = tuple(row[k] for k in indices)
        for i in range(len(indices), 0, -1):
            if idx[:i] in median_ages:
                return median_ages[idx[:i]]
        return median_ages[None]
    
    data['Age'] = data.apply(
        lambda row: find_median(row)
        if np.isnan(row['Age']) else row['Age'],
        axis=1
    )

    return data

```

#### Fare imputation

Fill with median of the same Pclass


```python
def impute_fare(data):
    train_data, _ = split_data(data)
    fare_indices = ['Pclass', 'Sex']
    # fare_indices = ['Pclass', 'Embarked']
    median_fares = train_data.dropna(subset=['Fare']).groupby(fare_indices).median()['Fare']
    median_fares = dict(zip(median_fares.index.to_flat_index(), median_fares.to_list()))

    data['Fare'] = data.apply(lambda row: median_fares[tuple(row[k] for k in fare_indices)] if np.isnan(row['Fare']) else row['Fare'], axis=1)

    return data
```

#### Cabin imputation

Just fill with a new value, 'U'


```python
"""
Some cabin values are weird:
- T
- D
- F
- F E69
- F G63

Some have multiple cabin values.
"""

def impute_cabin(data):
    data['Cabin'] = data['Cabin'].fillna(value='U')
    data['Deck'] = data['Cabin'].map(lambda d: d[0])
    return data
```

#### Embarked imputation

Fill with the most common value, 'S'


```python
def impute(data):
    print("Imputing age...")
    data = impute_age(data)
    
    print("Imputing fare...")
    data = impute_fare(data)
    
    print("Imputing cabin...")
    data = impute_cabin(data)
    
    print("Imputing embarked location...")
    data['Embarked'] = data['Embarked'].fillna(value='S')
    return data
```


```python
def preprocess(data, dummies=True, test=False):
    # get title and category
    # data['Title'] = data['Name'].map(get_title)
    # data['Category'] = data['Title'].map(get_category)
    print("adding title category...")
    data['Category'] = data['Name'].map(get_title).map(get_category)
    
    print("adding ticket features...")
    data = add_ticket_features(data)
    
    # Family info
    print("adding family features...")
    data = add_family_features(data)

    # combination features
    data['Sex&Pclass'] = data.apply(sex_and_class, axis=1)

    # impute?
    data = impute(data)
    
    print("adding cabin features...")
    data['CabinSurvived'] = add_cabin_survived(data)
    
    data['IsChild'] = data['Age'].map(is_child)
    data['MaleAdult'] = data.apply(is_male_adult, axis=1)

    # categorical to numerical
    data['Sex'] = data['Sex'].map(lambda s: 1 if s == 'male' else 0)
    
    dummy_vars = [
        'Embarked',
        #'Title',
        'Deck',
        'Pclass',
        'Category',
        'TicketNDigits',
        'TicketLetter',
        'FamilySurvived',
        'FamilyCategory',
        'TicketPrefix',
        'Sex&Pclass',
        'TicketSurvived',
        'CabinSurvived',
    ]
    if dummies:
        for dummy in dummy_vars:
            data = data.join(pd.get_dummies(data[dummy], prefix=dummy))
        data = data.drop(columns=dummy_vars)

    # remove unneeded columns
    if test:
        data['LastName'] = data['Name'].map(last_name)
        data = data.drop(columns=['Name'])
        data = data.drop(columns=['TicketNDigits', 'TicketLetter'])
    else:
        data = data.drop(columns=['Name', 'PassengerId'])
        data = data.drop(columns=['Cabin', 'Ticket'])
    print(data.shape)

    return data
```


```python
def split_data(data):
    # split into train and test
    return data.iloc[:891], data.iloc[891:]
```


```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
targets = train_data['Survived']
test_ids = test_data['PassengerId']
```


```python
data = train_data.append(test_data, sort=True)
data = data.reset_index()
print(data.shape)
```

    (1309, 13)
    


```python
data = train_data.append(test_data, sort=True)
data = data.reset_index()
train_X, test_X = split_data(preprocess(data, dummies=True))
train_X = train_X.drop(columns=['Survived', 'index'])
test_X = test_X.drop(columns=['Survived', 'index'])
```

    adding title category...
    adding ticket features...
    adding family features...
    Imputing age...
    Imputing fare...
    Imputing cabin...
    Imputing embarked location...
    adding cabin features...
    (1309, 79)
    


```python
pd.set_option('max_columns', 80)
train_X.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>FamilySize</th>
      <th>HaveFamily</th>
      <th>IsChild</th>
      <th>MaleAdult</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Deck_A</th>
      <th>Deck_B</th>
      <th>Deck_C</th>
      <th>Deck_D</th>
      <th>Deck_E</th>
      <th>Deck_F</th>
      <th>Deck_G</th>
      <th>Deck_T</th>
      <th>Deck_U</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Category_Master</th>
      <th>Category_Miss</th>
      <th>Category_Mr</th>
      <th>Category_Mrs</th>
      <th>Category_Officer</th>
      <th>Category_Other</th>
      <th>Category_Royalty</th>
      <th>TicketNDigits_1</th>
      <th>TicketNDigits_3</th>
      <th>TicketNDigits_4</th>
      <th>TicketNDigits_5</th>
      <th>TicketNDigits_6</th>
      <th>TicketNDigits_7</th>
      <th>TicketLetter_A</th>
      <th>TicketLetter_C</th>
      <th>TicketLetter_DIGIT</th>
      <th>TicketLetter_F</th>
      <th>TicketLetter_L</th>
      <th>TicketLetter_P</th>
      <th>TicketLetter_S</th>
      <th>TicketLetter_W</th>
      <th>FamilySurvived_N</th>
      <th>FamilySurvived_U</th>
      <th>FamilySurvived_Y</th>
      <th>FamilyCategory_Big</th>
      <th>FamilyCategory_Single</th>
      <th>FamilyCategory_Small</th>
      <th>TicketPrefix_A4</th>
      <th>TicketPrefix_A5</th>
      <th>TicketPrefix_C</th>
      <th>TicketPrefix_CA</th>
      <th>TicketPrefix_DIGIT</th>
      <th>TicketPrefix_FCC</th>
      <th>TicketPrefix_Other</th>
      <th>TicketPrefix_PC</th>
      <th>TicketPrefix_SCPARIS</th>
      <th>TicketPrefix_SOC</th>
      <th>TicketPrefix_SOTONOQ</th>
      <th>TicketPrefix_STONO</th>
      <th>TicketPrefix_STONO2</th>
      <th>TicketPrefix_WC</th>
      <th>Sex&amp;Pclass_female&amp;1</th>
      <th>Sex&amp;Pclass_female&amp;2</th>
      <th>Sex&amp;Pclass_female&amp;3</th>
      <th>Sex&amp;Pclass_male&amp;1</th>
      <th>Sex&amp;Pclass_male&amp;2</th>
      <th>Sex&amp;Pclass_male&amp;3</th>
      <th>TicketSurvived_N</th>
      <th>TicketSurvived_U</th>
      <th>TicketSurvived_Y</th>
      <th>CabinSurvived_N</th>
      <th>CabinSurvived_U</th>
      <th>CabinSurvived_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>26.0</td>
      <td>8.4583</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>54.0</td>
      <td>51.8625</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.0</td>
      <td>21.0750</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>27.0</td>
      <td>11.1333</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14.0</td>
      <td>30.0708</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Experiments


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
```


```python
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
```


```python
import xgboost as xgb
from xgboost import XGBClassifier, XGBRFClassifier
```

### Better validation sets

Default validation set during grid search is obtained by stratifiedKFold (so similar proportion of positive and negative classes as the full training set). But if the same last names/tickets appear in both the train and validation set, the validation results might not be as representative of test set performance.

So want to build better validation sets.

Idea 1: Split by last name, so different last names appear in train and val

Idea 2: Split by ticket prefix


```python
#split = StratifiedKFold(n_splits=5, shuffle=True)
last_names = train_data['Name'].map(last_name)
last_name_id = {
    name: i
    for i, name in enumerate(set(last_names))
}

ticket_prefixes = train_data['Ticket'].map(ticket_prefix)
ticket_id = {
    ticket: i
    for i, ticket in enumerate(set(ticket_prefixes))
}
map_id = {
    pair: i
    for i, pair in enumerate(set(zip(last_names, ticket_prefixes)))
}

groups = [
    #last_name_id[name] for name in last_names
    #ticket_id[ticket] for ticket in ticket_prefixes
    map_id[(name, prefix)] for name, prefix in zip(last_names, ticket_prefixes)
]

# pprint(map_id)
#pprint(Counter(groups))
#pprint(Counter(ticket_prefixes))

split = GroupKFold(n_splits=10)
for train, val in split.split(train_X, y=targets, groups=groups):
    #print(len(train), len(val))
    tr = train_data.iloc[train]
    va = train_data.iloc[val]
    
    # print(train)
    # print(val)
    
    tr_names = set(tr['Name'].map(last_name))
    va_names = set(va['Name'].map(last_name))
    
    tr_tickets = set(tr['Ticket'].map(ticket_prefix))
    va_tickets = set(va['Ticket'].map(ticket_prefix))
    """print(len(tr_names), len(va_names))
    print(len(tr_names & va_names))
    
    print(len(tr_tickets), len(va_tickets))
    print(len(tr_tickets & va_tickets))
    print()"""
```

### Grid search


```python
from sklearn.model_selection import GridSearchCV
```


```python
def output_gs_results(clf):
    print(clf.best_score_)
    pprint(clf.best_params_)
    results = clf.cv_results_
    best_min = None
    for mean, std, p in zip(results['mean_test_score'], results['std_test_score'], results['params']):
        print(f'{mean:.5f} (+/- {std:.5f}) for {p}')
        
        if best_min is None or mean - std > best_min[0]:
            best_min = mean - std, mean, std, p
            
    #pprint(best_min)
    print(best_min[0])
    print(f"{best_min[1]:.5f} (+/- {best_min[2]:.5f})")
    pprint(best_min[-1])
        
    return clf.best_params_

# cv_splitter = RepeatedStratifiedKFold(n_splits=10, shuffle=True)
# cv_splitter = StratifiedKFold(n_splits=10, shuffle=True)
cv_splitter = GroupKFold(n_splits=10)
```


```python
# class weight
n_pos = targets.sum()
n_neg = len(targets) - n_pos
print(n_neg, n_pos, n_neg / n_pos)
ratio = n_neg / n_pos
```

    549 342 1.605263157894737
    

### Random forest


```python
rf_params = [
    {
        'n_estimators': [25, 50, 100, 250],
        'max_features': ['sqrt', 'log2', 'auto'],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 4, 6, 10],
        'min_samples_leaf': [1, 3, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        #'bootstrap': [True, False]
    }
]
clf = GridSearchCV(RandomForestClassifier(),
                   rf_params,
                   #cv=cv_splitter,
                   cv=10,
                   iid=False,
                   scoring='accuracy',
                   n_jobs=10,
                   verbose=True)
clf.fit(train_X, targets, groups=groups)
best_params2 = output_gs_results(clf)
```


```python
xgbrf_params = [
    {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.1],
        'n_estimators': [25, 50, 100, 250],
        #'tree_method': ['exact', 'approx'],
        'objective': ['binary:logistic'],
        'min_child_weight': [1, 3, 10],
        'colsample_bytree': [0.5, 0.8, 1],
        'subsample': [0.5, 0.8, 1],
        'gamma': [0, 1e0, 1e1],
        'scale_pos_weight': [1, ratio],
        #'reg_alpha': [0, 1e-5, 1e-3],
        #'reg_lambda': [1e-5, 1e-1],
    }
]
clf = GridSearchCV(XGBRFClassifier(),
                   xgbrf_params,
                   scoring='accuracy',
                   #cv=cv_splitter,
                   cv=10,
                   iid=False,
                   n_jobs=10,
                   verbose=True)
clf.fit(train_X, targets, groups=groups)
best_params = output_gs_results(clf)
```

### Final training + Prediction


```python
def output_predictions(test_pred, test_ids):
    predictions = []

    for pred, idx in zip(test_pred, test_ids):
        predictions.append({'PassengerId': idx, 'Survived': pred})

    pred = pd.DataFrame(predictions)
    pred.to_csv('out.csv', sep=',', header=True, index=False)
```


```python
clf = RandomForestClassifier(
    max_depth=6,
    n_estimators=100,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=6,
    criterion='entropy',
    class_weight='balanced',
)
clf.fit(train_X, targets)
```


```python
clf = XGBRFClassifier(max_depth=4,
                      learning_rate=0.01,
                      n_estimators=25,
                      # reg_lambda=0,
                      gamma=0,
                      colsample_bytree=0.5,
                      subsample=0.5,
                      min_child_weight=1,
                      scale_pos_weight=1,
                      objective='binary:logistic')
clf.fit(train_X, targets)
```




    XGBRFClassifier(base_score=0.5, colsample_bylevel=1, colsample_bynode=0.8,
                    colsample_bytree=0.5, gamma=0, learning_rate=0.01,
                    max_delta_step=0, max_depth=4, min_child_weight=1, missing=None,
                    n_estimators=25, n_jobs=1, nthread=None,
                    objective='binary:logistic', random_state=0, reg_alpha=0,
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
                    subsample=0.5, verbosity=1)




```python
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
xgb.plot_importance(clf, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x270e82708c8>




![png](output_92_1.png)



```python
test_preds = clf.predict(test_X)
# print(test_preds[:5])
output_predictions(test_preds, test_ids)
```
