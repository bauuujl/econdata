# Analysis Portfolio

## Features
- **SQL** 
- **Power BI**

### Overview

I gathered the data from World Bank and FRED to build my portfolio where I used SQL to perform data analysis. I built two tables that contain the list of countries and indicators. Then, I used the two tables and joined them together with the values that correspond to each indicator and country.

I used Excel to combine various indicators and their values, matching them with specific country IDs to create mastersheet2.csv (utilizing VLOOKUP). 

*Also, please note that GDP data is nominal and the rates are not adjusted.*

### Results
#### Top 5 GDP in 2022
    SELECT 
        c.country_name, 
        iv.year, 
        iv.value, 
        i.indicator_name
    FROM indicator_values iv
    JOIN countries c ON iv.country_id = c.country_id
    JOIN indicators i ON iv.indicator_id = i.indicator_id
    WHERE iv.indicator_id = 1 
        AND iv.year = 2022
    ORDER BY iv.value DESC
    LIMIT 5;

1. *United States*
2. *China*
3. *Japan*
4. *Germany*
5. *India*

#### Top 40 High Income Countries by Inflation Rate in Ascending Order (2022)
    SELECT
	iv.year,
    c.country_name,
    c.incomegroup,
    iv.value,
    i.indicator_name
    FROM indicator_values iv
    JOIN countries c ON iv.country_id = c.country_id
    JOIN indicators i ON iv.indicator_id = i.indicator_id
    WHERE iv.indicator_id = 7 
	AND iv.year = 2022
    AND c.incomegroup LIKE '%High income%'
    Order by iv.value ASC
    LIMIT 40;

- *United States at No. 34 with 8.00%*

#### Top 40 High Income Countries by Inflation Rate in Ascending Order (2024)
    SELECT
	iv.year,
    c.country_name,
    c.incomegroup,
    iv.value,
    i.indicator_name
    FROM indicator_values iv
    JOIN countries c ON iv.country_id = c.country_id
    JOIN indicators i ON iv.indicator_id = i.indicator_id
    WHERE iv.indicator_id = 7 
	and iv.year = 2024
    AND c.incomegroup LIKE '%High income%'
    Order by iv.value ASC
    LIMIT 40;

- *United States at No. 36 with 2.95%*

### Visualization

I also created a view table for the US data so it's easier to populate if there are datas that need to be added.

    CREATE VIEW USdata AS
    SELECT
    iv.*,
    c.country_name,
    i.indicator_name
    FROM indicator_values iv
    JOIN countries c ON iv.country_id = c.country_id
    JOIN indicators i ON iv.indicator_id = i.indicator_id
    WHERE iv.country_id = 203;

![US GDP Growth Table](Screenshots/sqlecon.png)

Then, I visualize these datas using PowerBI.

![Analysis](Screenshots/powerbiecon.png)