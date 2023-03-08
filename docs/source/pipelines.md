For the documentation of mermaid visit https://mermaid-js.github.io/mermaid#/
# **Machine Learning Pipelines**

## Data Processing
```mermaid
graph LR
    raw[(Raw Data)] --> extract(Extract Model Data)
    extract --> clean(Clean)
    clean --> split(Split Data)
```

## Data Science
```mermaid
graph LR
    data_processing(Data Processing Pipeline) --> build(Build Model)
    build --> train(Train)
    train --> test(Test)
    test --> report[/Summary Report/]
    test --> visualization[/Visualizations/]
```

## Hyperparameter Optimization (HPO)
```mermaid
graph LR
    data_processing(Data Processing Pipeline) --> build(Build Model)
    build --> train(Train)
    train --> test(Test)
    test --> performance(Performance metrics)
    performance --> loop-ends{Search Ended?}
    loop-ends --> |No| change[change hyperparameter]
    change --> split
    loop-ends --> |Yes| return-best[Take best model]
    return-best --> report[/Summary Report/]


    subgraph loop[Search Space Loop]
        split
        build
        train
        test
        performance
        loop-ends
        change
    end
```