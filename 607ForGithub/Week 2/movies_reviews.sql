
DROP SCHEMA IF EXISTS movies;

CREATE SCHEMA movies;

DROP TABLE IF EXISTS movies.movie_reviews;

CREATE TABLE movies.movie_reviews (
  movie_names VARCHAR(255) NOT NULL,
  rating INT NOT NULL,
  rater VARCHAR(30) NOT NULL
  );
  
LOAD DATA INFILE '/Users/euniceok/PycharmProjects/cuny/spring2019/Week2/hw/data/movies_reviews.csv' 
INTO TABLE movies.movie_reviews 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r'
IGNORE 1 LINES
(movie_names,rating,rater);

-- SELECT * FROM movies.movie_reviews;