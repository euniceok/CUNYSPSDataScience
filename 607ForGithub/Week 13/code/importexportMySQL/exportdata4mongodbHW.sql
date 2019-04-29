SELECT *
FROM mongodbHW.indiacitiespop
INTO OUTFILE '/Users/euniceok/PycharmProjects/cuny/spring2019/Week13DB/output/IndiaCitiesPopulation.csv' 
FIELDS ENCLOSED BY '"' 
TERMINATED BY ';' 
ESCAPED BY '"' 
LINES TERMINATED BY '\r\n';