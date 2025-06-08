CREATE DATABASE IF NOT EXISTS economic_analysis;

USE economic_analysis;

-----

CREATE TABLE countries (
    country_id INT AUTO_INCREMENT PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    country_code CHAR(3)
);

CREATE TABLE indicators (
    indicator_id INT AUTO_INCREMENT PRIMARY KEY,
    indicator_name VARCHAR(255) NOT NULL,
    indicator_code VARCHAR(50)
);

CREATE TABLE indicator_values (
    value_id INT AUTO_INCREMENT PRIMARY KEY,
    country_id INT NOT NULL,
    indicator_id INT NOT NULL,
    year YEAR NOT NULL,
    value DECIMAL(18, 4) NOT NULL,
    gender VARCHAR(10),
    FOREIGN KEY (country_id) REFERENCES countries(country_id),
    FOREIGN KEY (indicator_id) REFERENCES indicators(indicator_id)
);

-- Allow Local File
SET GLOBAL local_infile = 1;

--

ALTER TABLE countries
ADD COLUMN region VARCHAR(50),
ADD COLUMN incomegroup VARCHAR(50);

-- Import countries

LOAD DATA LOCAL INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/countries.csv'
INTO TABLE countries
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(country_name, country_code, region, incomegroup);

DELETE FROM indicator_values;
ALTER TABLE indicator_values AUTO_INCREMENT = 1;
DELETE FROM indicator_values WHERE 1=1;

-- Import Indicators

LOAD DATA LOCAL INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/indicators.csv'
INTO TABLE indicators
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(indicator_name);

ALTER TABLE indicators
DROP COLUMN indicator_code;

-- Import the values for countries and indicators

LOAD DATA LOCAL INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/mastersheet2.csv'
INTO TABLE indicator_values
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(country_id, indicator_id, year, value, gender);

SELECT * FROM indicator_values
WHERE indicator_id = 4;

--

SHOW TABLES;
DESCRIBE indicator_values;
SELECT * FROM indicator_values LIMIT 10;

-- Load first 30 (alphabetically)

SELECT
  iv.*,
  c.country_name,
  i.indicator_name
FROM
  indicator_values iv
JOIN countries c ON iv.country_id = c.country_id
JOIN indicators i ON iv.indicator_id = i.indicator_id
LIMIT 30;

-- Shows US GDP in chronological order 

SELECT c.country_name,
year, 
value
FROM countries c
JOIN indicator_values iv ON iv.country_id = c.country_id
WHERE c.country_name = 'United States' AND iv.indicator_id = 1
ORDER BY year;

-- Shows GDP per country from highest to lowest

SELECT c.country_name, iv.year, iv.value, i.indicator_name
FROM indicator_values iv
JOIN countries c ON iv.country_id = c.country_id
JOIN indicators i ON iv.indicator_id = i.indicator_id
WHERE iv.indicator_id = 1 AND iv.year = 2022
ORDER BY iv.value DESC;

-- GDP Growth

SELECT
  c.country_name,
  a.country_id,
  a.year,
  a.value AS this_year,
  b.value AS last_year,
  ((a.value - b.value) / b.value) * 100 AS yoy_growth
FROM
  indicator_values a
JOIN indicator_values b
  ON a.country_id = b.country_id
  AND a.indicator_id = b.indicator_id
  AND a.year = b.year + 1
JOIN countries c ON c.country_id = a.country_id
WHERE a.indicator_id = 1
AND c.country_name = 'United States'
ORDER BY a.country_id, a.year;

-- Create a view table for US data

CREATE VIEW gdp_us_trend AS
SELECT year, value
FROM indicator_values
WHERE country_id = 1 AND indicator_id = 1;

DROP VIEW IF EXISTS gdp_us_trend;

CREATE VIEW USdata AS
SELECT
    iv.*,
    c.country_name,
    i.indicator_name
FROM indicator_values iv
JOIN countries c ON iv.country_id = c.country_id
JOIN indicators i ON iv.indicator_id = i.indicator_id
WHERE iv.country_id = 203;
  
SELECT * FROM USdata;

-- Shows US GDP Growth using windows function LAG

SELECT 
    year,
    value AS gdp,
    LAG(value) OVER (ORDER BY year) AS prev_gdp,
    ROUND((value - LAG(value) OVER (ORDER BY year)) / LAG(value) OVER (ORDER BY year) * 100, 2) AS US_gdp_growth
FROM USdata
WHERE indicator_name = 'GDP (current US$)'
ORDER BY year; -- United States GDP Growth
