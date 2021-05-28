# Streamlit101

This streamlit app uses Prophet and is deployed in a docker container and has the intention for a blueprint to use AWS app runner.


## 1. Data

* Choose and prepare data.
* I learned that open data in germany provide some interesting data for real time 
  mickey mouse examples. So I choose one with counting points for bicycles and add 
  some weather data.
  
* See the notebook for some inspiration.


## 2. Dockerfile

* To use it in proper way you can build your docker image.
* Github action makes it easy to import the image to AWS ECR for further usage in
EC2, Fargate or AppRunner.
  

## 3. Streamlit

* Streamlit is an ideal tool to visualize data-science stuff quickly and using as background
for discussion or a low cost prototyping environment.