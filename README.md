
# Text Classification
In this project, we used 3 different metrics (Information Gain, Mutual Information, Chi Squared) to find important words and then we used them for the classification task. We compared the result at the end. <br>

To read jupyter notebook complete code click on [Codes](Codes/README.md)


Each document can be represented by the set of its words. <br>
But some words are more important and has more effect and more meaning. These words can be used for determining the context of a document. <br>
In this part, we are going to find a set of 100 words that are more infomative for __document classification__. <br>


## Data Set
The dataset for this task is "همشهری" (Hamshahri) that contains 8600 persian documents. 


```python
class_name
```




    ['ورزش', 'اقتصاد', 'ادب و هنر', 'اجتماعی', 'سیاسی']



There is some statistcs information about out dataset.




    vocab size: 65880
    number of terms (all tokens): 3506727
    number of docs: 8600
    number of classes: 5
    

The probability of each classes are stored in the table below.







<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.232558</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.255814</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.058140</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.209302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.244186</td>
    </tr>
  </tbody>
</table>
</div>



   # Calculation  of our metrics for each words in Vocabulary
   We are going to find 100 words that are good indicator of classes. <br>
   We want to use 3 different type of metrics  
   > -  Information Gain  
   > -  Mutual Information  
   > -  $\chi$ square   
   
   

## Information Gain
The top 10 words with highest information gain can be seen in the table below.







<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>information_gain</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.612973</td>
      <td>ورزشی</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.516330</td>
      <td>تیم</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.297086</td>
      <td>اجتماعی</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.293313</td>
      <td>سیاسی</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.283891</td>
      <td>فوتبال</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.267878</td>
      <td>اقتصادی</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.225276</td>
      <td>بازی</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.223381</td>
      <td>جام</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.197755</td>
      <td>قهرمانی</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.177807</td>
      <td>اسلامی</td>
    </tr>
  </tbody>
</table>
</div>



If you think about the meaning of these words you can agree that since they a have high information gain they can be really good identifiers for categorizing a doc. <br>
In the table below, you can see 5 worst words.







<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>information_gain</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65875</th>
      <td>0.000042</td>
      <td>تبهکاران</td>
    </tr>
    <tr>
      <th>65876</th>
      <td>0.000042</td>
      <td>پذیرفتم</td>
    </tr>
    <tr>
      <th>65877</th>
      <td>0.000041</td>
      <td>توقف</td>
    </tr>
    <tr>
      <th>65878</th>
      <td>0.000031</td>
      <td>سلمان</td>
    </tr>
    <tr>
      <th>65879</th>
      <td>0.000027</td>
      <td>چهارمحال</td>
    </tr>
  </tbody>
</table>
</div>



## Mutual Information
We found two formulas for this metrics. The first one only calculates the MI in the case that w = 1, ci = 1. The second one calculates all 4 possible combinations of (w, ci) and multiplies them by their probabilities.  <br>
We used both of them and then choose the better set. <br> 
Let's see the result of each of those different formulas.

### First Formula   
The formula is:

![formula-1](f1.bmp)



The result for this formula






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mutual information(MI)</th>
      <th>main class MI</th>
      <th>main_class</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.524418</td>
      <td>1.782409</td>
      <td>ادب و هنر</td>
      <td>مایکروسافت</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.524418</td>
      <td>1.782409</td>
      <td>ادب و هنر</td>
      <td>باغات</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.524418</td>
      <td>1.782409</td>
      <td>ادب و هنر</td>
      <td>نباریدن</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.524191</td>
      <td>1.703799</td>
      <td>اقتصاد</td>
      <td>آلیاژی</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.524191</td>
      <td>1.703799</td>
      <td>اقتصاد</td>
      <td>دادمان</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.524191</td>
      <td>1.703799</td>
      <td>اقتصاد</td>
      <td>فرازها</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.524191</td>
      <td>1.703799</td>
      <td>اقتصاد</td>
      <td>سیاتل</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.521680</td>
      <td>1.782409</td>
      <td>ادب و هنر</td>
      <td>سخیف</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.521680</td>
      <td>1.782409</td>
      <td>ادب و هنر</td>
      <td>کارناوال</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.521680</td>
      <td>1.782409</td>
      <td>ادب و هنر</td>
      <td>تحتانی</td>
    </tr>
  </tbody>
</table>
</div>



#### <font color=red>But there is a __problem__ here  
The words in the above table are __infrequent__  words.<br>
They can give us high information in the case that they appear in our text. But when they __do not appear__ in our document, the absence of those words __do not give__ us any information about the class. So these words are only useful for those documents that have these words. So for most of the documents, these words are not useful.<br>
It seems like it is important to consider the frequency of different cases of (w, ci) and also its mutual information. <br>
    
You can see the number of occurrences of some of these words in each class:


```python
word_occurance_frequency_vs_class[word_index['نباریدن']], word_occurance_frequency_vs_class[word_index['مایکروسافت']],  word_occurance_frequency_vs_class[word_index['آلیاژی']]
```




    (array([0, 4, 1, 0, 0]), array([0, 4, 1, 0, 0]), array([0, 5, 1, 0, 0]))






### Second Model
In the second model, the above problem is solved because we multiplied the frequency of different cases (probability) by its mutual information. 

The Second formula is:

![formula-2](f2.bmp)










<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mutual information(MI)</th>
      <th>main class MI</th>
      <th>main_class</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.202674</td>
      <td>0.606665</td>
      <td>ورزش</td>
      <td>ورزشی</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.173929</td>
      <td>0.512590</td>
      <td>ورزش</td>
      <td>تیم</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.099833</td>
      <td>0.279402</td>
      <td>ورزش</td>
      <td>فوتبال</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.094818</td>
      <td>0.258578</td>
      <td>سیاسی</td>
      <td>سیاسی</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.088517</td>
      <td>0.232945</td>
      <td>اقتصاد</td>
      <td>اقتصادی</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.085858</td>
      <td>0.265246</td>
      <td>اجتماعی</td>
      <td>اجتماعی</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.076582</td>
      <td>0.209092</td>
      <td>ورزش</td>
      <td>بازی</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.076098</td>
      <td>0.217883</td>
      <td>ورزش</td>
      <td>جام</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.070387</td>
      <td>0.195426</td>
      <td>ورزش</td>
      <td>قهرمانی</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.056253</td>
      <td>0.155666</td>
      <td>ورزش</td>
      <td>بازیکن</td>
    </tr>
  </tbody>
</table>
</div>



#### These words are better
They are not __rare__ <br> 
You can see the frequency of some of these words in each classes:


```python
print (list(reversed(class_name)))
word_occurance_frequency_vs_class[word_index['ورزشی']], word_occurance_frequency_vs_class[word_index['سیاسی']],  word_occurance_frequency_vs_class[word_index['پیروزی']],
```

    ['سیاسی', 'اجتماعی', 'ادب و هنر', 'اقتصاد', 'ورزش']
    




    (array([1866,    9,    8,   66,   16]),
     array([  35,  300,   71,  372, 1602]),
     array([497,  44,  24,  60, 191]))



## $ \chi$ Squared
Another Metric is $ \chi$ Squared. <br>
Now we are goind to see the result of using $\chi$ squared as a mesure of importance.







<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chi squared</th>
      <th>main class chi</th>
      <th>main_class</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2234.591600</td>
      <td>7376.515420</td>
      <td>ورزش</td>
      <td>ورزشی</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2028.186959</td>
      <td>6701.136193</td>
      <td>ورزش</td>
      <td>تیم</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1302.947803</td>
      <td>4296.622009</td>
      <td>ورزش</td>
      <td>فوتبال</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1050.289281</td>
      <td>3459.737651</td>
      <td>ورزش</td>
      <td>جام</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1043.478941</td>
      <td>3138.957039</td>
      <td>سیاسی</td>
      <td>سیاسی</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1029.481451</td>
      <td>3055.759330</td>
      <td>اقتصاد</td>
      <td>اقتصادی</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1027.311167</td>
      <td>3254.137538</td>
      <td>ورزش</td>
      <td>بازی</td>
    </tr>
    <tr>
      <th>7</th>
      <td>967.923254</td>
      <td>3299.353018</td>
      <td>اجتماعی</td>
      <td>اجتماعی</td>
    </tr>
    <tr>
      <th>8</th>
      <td>961.078769</td>
      <td>3169.912598</td>
      <td>ورزش</td>
      <td>قهرمانی</td>
    </tr>
    <tr>
      <th>9</th>
      <td>772.041205</td>
      <td>2558.041173</td>
      <td>ورزش</td>
      <td>بازیکن</td>
    </tr>
  </tbody>
</table>
</div>



## Result Comparison
We can compare our three set of words here. <br>
In the table below, you can see top 20 words for each metrics.<br>
For mutual information, there are two sets because we used two different formula.







<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>information gain1</th>
      <th>chi squared1</th>
      <th>mutual information_model_2</th>
      <th>mutual information_model_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ورزشی</td>
      <td>ورزشی</td>
      <td>ورزشی</td>
      <td>مایکروسافت</td>
    </tr>
    <tr>
      <th>1</th>
      <td>تیم</td>
      <td>تیم</td>
      <td>تیم</td>
      <td>باغات</td>
    </tr>
    <tr>
      <th>2</th>
      <td>اجتماعی</td>
      <td>فوتبال</td>
      <td>فوتبال</td>
      <td>نباریدن</td>
    </tr>
    <tr>
      <th>3</th>
      <td>سیاسی</td>
      <td>جام</td>
      <td>سیاسی</td>
      <td>آلیاژی</td>
    </tr>
    <tr>
      <th>4</th>
      <td>فوتبال</td>
      <td>سیاسی</td>
      <td>اقتصادی</td>
      <td>دادمان</td>
    </tr>
    <tr>
      <th>5</th>
      <td>اقتصادی</td>
      <td>اقتصادی</td>
      <td>اجتماعی</td>
      <td>فرازها</td>
    </tr>
    <tr>
      <th>6</th>
      <td>بازی</td>
      <td>بازی</td>
      <td>بازی</td>
      <td>سیاتل</td>
    </tr>
    <tr>
      <th>7</th>
      <td>جام</td>
      <td>اجتماعی</td>
      <td>جام</td>
      <td>سخیف</td>
    </tr>
    <tr>
      <th>8</th>
      <td>قهرمانی</td>
      <td>قهرمانی</td>
      <td>قهرمانی</td>
      <td>کارناوال</td>
    </tr>
    <tr>
      <th>9</th>
      <td>اسلامی</td>
      <td>بازیکن</td>
      <td>بازیکن</td>
      <td>تحتانی</td>
    </tr>
    <tr>
      <th>10</th>
      <td>بازیکن</td>
      <td>بازیکنان</td>
      <td>اسلامی</td>
      <td>معراج</td>
    </tr>
    <tr>
      <th>11</th>
      <td>مجلس</td>
      <td>فدراسیون</td>
      <td>بازیکنان</td>
      <td>خلافت</td>
    </tr>
    <tr>
      <th>12</th>
      <td>بازیکنان</td>
      <td>مسابقات</td>
      <td>فدراسیون</td>
      <td>نجیب</td>
    </tr>
    <tr>
      <th>13</th>
      <td>فدراسیون</td>
      <td>دلار</td>
      <td>مسابقات</td>
      <td>زبون</td>
    </tr>
    <tr>
      <th>14</th>
      <td>مسابقات</td>
      <td>قیمت</td>
      <td>دلار</td>
      <td>صحیفه</td>
    </tr>
    <tr>
      <th>15</th>
      <td>شورای</td>
      <td>مسابقه</td>
      <td>مسابقه</td>
      <td>مشمئزکننده</td>
    </tr>
    <tr>
      <th>16</th>
      <td>مسابقه</td>
      <td>آسیا</td>
      <td>مجلس</td>
      <td>ارتجاع</td>
    </tr>
    <tr>
      <th>17</th>
      <td>دلار</td>
      <td>گذاری</td>
      <td>قیمت</td>
      <td>ذبیح</td>
    </tr>
    <tr>
      <th>18</th>
      <td>آسیا</td>
      <td>صنایع</td>
      <td>شورای</td>
      <td>وصنایع</td>
    </tr>
    <tr>
      <th>19</th>
      <td>مردم</td>
      <td>سرمایه</td>
      <td>گذاری</td>
      <td>توپخانه</td>
    </tr>
  </tbody>
</table>
</div>



## Output File
The output files are stored in CSV format files. <br>
These files contain 100 most important words for each metrics.

## Conclusion of Part 1 (Finding important words for classification)
When we look at the last table we can see first three columns are similar to each other and have nearly the same words but last columns' (mutual information with formula 1) words differse from other columns <br>
We can conclude that the Formula 1 has different behavior and probably is not efficient. So it is better to use formula 2 to calculate Mutual Information <br>
<br>
For more accurate comparisons on which of these three metrics are better, we can test it.<br>
#### In part 2 we are going to test which metric is better, with a classification task.

# Part 2 (Classifying using words in part 1)
In Part 1 we tried to find good features to vectorize documents. We used three metrics and extracted three set of 100 words. <br>
Each document can be represented by the set of words that appear in the document. <br>
In this part we want to use these sets of features to classify documents with __SVM__. <br>




## Evaluation
To evaluate our classification we used k-fold cross-validation with k=5. <br>
We reported our average of these 5 confusion matrices.

   ## Vectorizing Documents
   We wanted to vectorize our documents. We used 4 different methods:
   > 1) Using 1000 most frequent words as features set <br>
   > 2) Using Information Gain features <br>
   > 3) Using Mutual Information features <br>
   > 4) Using $\chi$ square   features <br>
   
   

## 1) Using 1000 most frequent words as feature set
There is an ambiguity in defining the meaning of frequent. <br>
> -  First meaning: A word is frequent if in lots of document there is at least one occurrence of this word. <br>
> -  Second meaning: A word is frequent if the sum of the number of occurrence of this word in all documents is high. (Maybe in one document there are lots of occurrences but in another document, there is no occurrence.) <br> 

In this code, we chose the __first meaning__.








<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8415</td>
      <td>و</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8352</td>
      <td>در</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8241</td>
      <td>به</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7956</td>
      <td>از</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7838</td>
      <td>این</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7382</td>
      <td>با</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7240</td>
      <td>که</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6923</td>
      <td>را</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6912</td>
      <td>است</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6859</td>
      <td>می</td>
    </tr>
  </tbody>
</table>
</div>



The above words are 10  most frequent words in our dataset. And all of them are stop words.
    

### Making Vector X
We want to make the vector for each document and then use this vectors for classification. <br>
We used our 1000 words for vectorizing.

### Using SVM for classification 

We used svm classifier for our classification.

### Confusion Matrix for 1000 most frequent




    accuracy: 0.8787209302325582 
    
    confusion matrix:
    
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>389.0</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>6.4</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2</td>
      <td>414.2</td>
      <td>0.2</td>
      <td>6.4</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.2</td>
      <td>10.8</td>
      <td>37.2</td>
      <td>44.2</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.6</td>
      <td>18.4</td>
      <td>1.6</td>
      <td>300.6</td>
      <td>37.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>19.2</td>
      <td>0.0</td>
      <td>29.4</td>
      <td>370.4</td>
    </tr>
  </tbody>
</table>
</div>



## 2) Using 100-dimensional vector with Information Gain 


    







<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>information_gain</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.612973</td>
      <td>ورزشی</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.516330</td>
      <td>تیم</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.297086</td>
      <td>اجتماعی</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.293313</td>
      <td>سیاسی</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.283891</td>
      <td>فوتبال</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.267878</td>
      <td>اقتصادی</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.225276</td>
      <td>بازی</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.223381</td>
      <td>جام</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.197755</td>
      <td>قهرمانی</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.177807</td>
      <td>اسلامی</td>
    </tr>
  </tbody>
</table>
</div>



### Making Vector X
We made vector for each document and then we used this vectors for classification with SVM.




    accuracy: 0.806279069767442 
    
    confusion matrix:
    
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>376.0</td>
      <td>7.6</td>
      <td>0.2</td>
      <td>11.4</td>
      <td>4.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.4</td>
      <td>382.6</td>
      <td>0.2</td>
      <td>22.0</td>
      <td>33.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>24.0</td>
      <td>7.2</td>
      <td>55.0</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.4</td>
      <td>30.8</td>
      <td>1.0</td>
      <td>268.8</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.4</td>
      <td>31.6</td>
      <td>0.2</td>
      <td>33.6</td>
      <td>352.2</td>
    </tr>
  </tbody>
</table>
</div>



## 3) Using 100-dimensional vector with Mutal Information
We used the better formula for selecting 100 words.









<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mutual information</th>
      <th>main class score</th>
      <th>main class</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.202674</td>
      <td>0.606665</td>
      <td>ورزش</td>
      <td>ورزشی</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.173929</td>
      <td>0.512590</td>
      <td>ورزش</td>
      <td>تیم</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.099833</td>
      <td>0.279402</td>
      <td>ورزش</td>
      <td>فوتبال</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.094818</td>
      <td>0.258578</td>
      <td>سیاسی</td>
      <td>سیاسی</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.088517</td>
      <td>0.232945</td>
      <td>اقتصاد</td>
      <td>اقتصادی</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.085858</td>
      <td>0.265246</td>
      <td>اجتماعی</td>
      <td>اجتماعی</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.076582</td>
      <td>0.209092</td>
      <td>ورزش</td>
      <td>بازی</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.076098</td>
      <td>0.217883</td>
      <td>ورزش</td>
      <td>جام</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.070387</td>
      <td>0.195426</td>
      <td>ورزش</td>
      <td>قهرمانی</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.056253</td>
      <td>0.155666</td>
      <td>ورزش</td>
      <td>بازیکن</td>
    </tr>
  </tbody>
</table>
</div>






    accuracy: 0.804186046511628 
    
    confusion matrix:
    
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>375.2</td>
      <td>8.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.4</td>
      <td>379.8</td>
      <td>0.2</td>
      <td>22.6</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.8</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>56.4</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>31.4</td>
      <td>1.0</td>
      <td>270.0</td>
      <td>55.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.4</td>
      <td>31.2</td>
      <td>0.2</td>
      <td>35.0</td>
      <td>351.2</td>
    </tr>
  </tbody>
</table>
</div>



## 4) Using 100-dimensional vector with $\chi$ Squared










<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chi squared</th>
      <th>main class score</th>
      <th>main class</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2234.591600</td>
      <td>7376.515420</td>
      <td>ورزش</td>
      <td>ورزشی</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2028.186959</td>
      <td>6701.136193</td>
      <td>ورزش</td>
      <td>تیم</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1302.947803</td>
      <td>4296.622009</td>
      <td>ورزش</td>
      <td>فوتبال</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1050.289281</td>
      <td>3459.737651</td>
      <td>ورزش</td>
      <td>جام</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1043.478941</td>
      <td>3138.957039</td>
      <td>سیاسی</td>
      <td>سیاسی</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1029.481451</td>
      <td>3055.759330</td>
      <td>اقتصاد</td>
      <td>اقتصادی</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1027.311167</td>
      <td>3254.137538</td>
      <td>ورزش</td>
      <td>بازی</td>
    </tr>
    <tr>
      <th>7</th>
      <td>967.923254</td>
      <td>3299.353018</td>
      <td>اجتماعی</td>
      <td>اجتماعی</td>
    </tr>
    <tr>
      <th>8</th>
      <td>961.078769</td>
      <td>3169.912598</td>
      <td>ورزش</td>
      <td>قهرمانی</td>
    </tr>
    <tr>
      <th>9</th>
      <td>772.041205</td>
      <td>2558.041173</td>
      <td>ورزش</td>
      <td>بازیکن</td>
    </tr>
  </tbody>
</table>
</div>






    accuracy: 0.8025581395348838 
    
    confusion matrix:
    
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>375.2</td>
      <td>8.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.4</td>
      <td>380.8</td>
      <td>0.2</td>
      <td>22.2</td>
      <td>35.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.8</td>
      <td>25.0</td>
      <td>6.6</td>
      <td>55.6</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.4</td>
      <td>31.2</td>
      <td>0.8</td>
      <td>267.2</td>
      <td>58.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.4</td>
      <td>31.8</td>
      <td>0.2</td>
      <td>35.0</td>
      <td>350.6</td>
    </tr>
  </tbody>
</table>
</div>



## Comparision
We compare our result with these 4 methods with confusion matrix and accuracy. <br>
The result is as follows.




    accuracy:
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1000words</th>
      <th>InfoGain</th>
      <th>chi squared</th>
      <th>mutual info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.878721</td>
      <td>0.806279</td>
      <td>0.802558</td>
      <td>0.804186</td>
    </tr>
  </tbody>
</table>
</div>




```python
preview = pd.concat([pd.DataFrame(first_method_confusion_matrix),       
                     pd.DataFrame(second_method_confusion_matrix),
                     pd.DataFrame(third_method_confusion_matrix),
                     pd.DataFrame(forth_method_confusion_matrix)], axis=1)
print ("confusion matrix:")
print ("\t1000 words\t\t\t IG \t\t \t MI \t\t   chi squared")
preview
```

    confusion matrix:
    	1000 words			 IG 		 	 MI 		   chi squared
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>389.0</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>6.4</td>
      <td>2.4</td>
      <td>376.0</td>
      <td>7.6</td>
      <td>0.2</td>
      <td>11.4</td>
      <td>4.8</td>
      <td>375.2</td>
      <td>8.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>5.2</td>
      <td>375.2</td>
      <td>8.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2</td>
      <td>414.2</td>
      <td>0.2</td>
      <td>6.4</td>
      <td>19.0</td>
      <td>1.4</td>
      <td>382.6</td>
      <td>0.2</td>
      <td>22.0</td>
      <td>33.8</td>
      <td>1.4</td>
      <td>379.8</td>
      <td>0.2</td>
      <td>22.6</td>
      <td>36.0</td>
      <td>1.4</td>
      <td>380.8</td>
      <td>0.2</td>
      <td>22.2</td>
      <td>35.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.2</td>
      <td>10.8</td>
      <td>37.2</td>
      <td>44.2</td>
      <td>6.6</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>7.2</td>
      <td>55.0</td>
      <td>9.8</td>
      <td>3.8</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>56.4</td>
      <td>9.8</td>
      <td>3.8</td>
      <td>25.0</td>
      <td>6.6</td>
      <td>55.6</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.6</td>
      <td>18.4</td>
      <td>1.6</td>
      <td>300.6</td>
      <td>37.8</td>
      <td>2.4</td>
      <td>30.8</td>
      <td>1.0</td>
      <td>268.8</td>
      <td>57.0</td>
      <td>2.0</td>
      <td>31.4</td>
      <td>1.0</td>
      <td>270.0</td>
      <td>55.6</td>
      <td>2.4</td>
      <td>31.2</td>
      <td>0.8</td>
      <td>267.2</td>
      <td>58.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>19.2</td>
      <td>0.0</td>
      <td>29.4</td>
      <td>370.4</td>
      <td>2.4</td>
      <td>31.6</td>
      <td>0.2</td>
      <td>33.6</td>
      <td>352.2</td>
      <td>2.4</td>
      <td>31.2</td>
      <td>0.2</td>
      <td>35.0</td>
      <td>351.2</td>
      <td>2.4</td>
      <td>31.8</td>
      <td>0.2</td>
      <td>35.0</td>
      <td>350.6</td>
    </tr>
  </tbody>
</table>
</div>



## Visualization

We are going to show each one in separated tables:


```python
# plt.figure()
# plot_confusion_matrix(first_method_cm, classes=class_name,
#                       title='Confusion matrix visualization for chi squared');
plt.figure()
plot_confusion_matrix(first_method_cm, classes=class_name,
                      title='Confusion matrix visualization for 1000 most frequent_normalized', normalize=True);
plt.show()
```


![png](README/output_62_0.png)



```python
plt.figure()
plot_confusion_matrix(second_method_cm, classes=class_name,
                      title='Confusion matrix visualization for Information gain_normalized', normalize=True);
plt.show()
```


![png](README/output_63_0.png)



```python
plt.figure()
plot_confusion_matrix(third_method_cm, classes=class_name,
                      title='Confusion matrix visualization for Mutual information_normalized', normalize=True);
plt.show()
```


![png](README/output_64_0.png)



```python
plt.figure()
plot_confusion_matrix(forth_method_cm, classes=class_name,
                      title='Confusion matrix visualization for chi squared_normalized', normalize=True);
plt.show()
```


![png](README/output_65_0.png)


#### Now we want to test the result for most 100 frequent word vectors
Because maybe this is not fair to compare 1000 frequent words with 100 words in other metrics.

#### Confusion Matrix for 100 most frequent



    accuracy: 0.803953488372093 
    
    confusion matrix:
    
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>375.6</td>
      <td>8.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>4.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.4</td>
      <td>382.4</td>
      <td>0.4</td>
      <td>22.0</td>
      <td>33.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>24.8</td>
      <td>7.2</td>
      <td>55.2</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.8</td>
      <td>30.8</td>
      <td>1.0</td>
      <td>267.4</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.4</td>
      <td>32.6</td>
      <td>0.2</td>
      <td>34.6</td>
      <td>350.2</td>
    </tr>
  </tbody>
</table>
</div>







![png](README/output_69_0.png)


## Conclusion
We know that 1000 words are not best words because these words include stop words that they are not informative. But because they are 1000 words rather than 100 words, the result is going to be better. <br>
We test a set of 100 most frequent words. The result was acc = 0.8049 which is similar to other three methods. We also show the confusion matrix for 100 most frequent in the last table above. And you can see this table is also similar to other three ones <br>
So we can guess that there is no significant difference between choosing these metrics for selecting words in document classification task. <br> 
And we also know that Information Gain doesn't store every class information gains (We only stored one number Information Gain for every word). But we can consider each class information gain if we split the Sigma over classes in information gain formula.(And consider the meaning of Entropy for each class). So it has the same functionality as other metrics. <br>
We guess that if the __dimension of our vector__ increase we probably are going to get more accuracy.<br>
And also maybe if we use different features depending on __the sequence of words__ that appear each document, we can get more accuracy.
