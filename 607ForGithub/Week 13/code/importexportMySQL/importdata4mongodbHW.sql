
DROP SCHEMA IF EXISTS mongodbHW;

CREATE SCHEMA mongodbHW;

DROP TABLE IF EXISTS mongodbHW.indiacitiespop;

CREATE TABLE mongodbHW.indiacitiespop (
  _id INT NOT NULL,
  state VARCHAR(255) NOT NULL,
  city VARCHAR(255) NOT NULL,
  population INT NOT NULL,
  literates INT NOT NULL,
  sexRatio INT NOT NULL,
  isMetro VARCHAR(255) NULL
  );
  
LOAD DATA INFILE '/Users/euniceok/PycharmProjects/cuny/spring2019/Week13DB/data/IndiaCitiesPopulation.csv' 
INTO TABLE mongodbHW.indiacitiespop 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r'
IGNORE 1 LINES
(_id,state,city,population,literates,sexRatio,isMetro);