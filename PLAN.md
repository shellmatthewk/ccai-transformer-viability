The plan is to submit to CCAI Workshop (NeurIPS 2026). It gives us about 5 weeks, which I think is doable as we only have room for 4 pages. I do not have the brain spcae to do this myself, but I think this is valuable for the future as roles can be hiring in this space in the future that we may be interested in. 

*When Does Learned Climate Field Reconstruction Fail? A Controlled Study of Spatial Transfer and Extreme Events*

This is the current plan. It was AI generated. do not come for me. Let's break it down: 

- Climate Field Reconstruction: https://gmd.copernicus.org/articles/17/3409/2024/;
This is the estimation of spatiotemporal climate fields, which is spatial + time series data, which is time series data within multiple objects, essentially. 
- What we want: when does this reconstruction fail? (at what point does it become unreliable?)
- A controlled study of spatial transfer and extreme events
Study of if a model generalizes across space or only where it was trained from or within extreme weather or climate events?

We know that transformers, an architecture of neural networks can learn climate maps and stuff and predict weather, heat, etc. But, do these work the same at other stations/regions? And do they also work during climate events? 

We add conformal prediction to add uncertainty predictions, so that we can state if uncertainty is a warning or a false positive? 

# So what are we doing? 

There are approximately 3,000 climate weather stations in Europe that measure rainfall, temperature, humidity, all that stuff. We currently have frameworks that can take all of these weather stations and essentially fill in a map of all the spaces connected and predict the temperature / climate of that spot based on the pattersn from the weather stations. (think like, voronoi diagrams or something) 

We train the model based on, lets say Western Europe temperatures, then use whatever frameworks to verify the map termperatures. We then use this same trained model on eastern europe readings (no training) and compare to the actual ground truth tremperature versus what the model predicts. 

# Why does it matter? 

Climate scientists need cheaper models, and they need reliable methods for prediction and being able to trust said model(s). We want to be able to state which method of network fails the least (or the most successful) during spatial transfer? 

When these models do fail, does uncertainty warn, or is it confidently wrong? 
Does calibrating uncertainty locally help when moving to a new region? 

So if the these networks can't even predict across Europe that have weather stations that are so densely populated across the continent, how would it do in Africa? South America? 

There is a method called Kriging - which is a stats method from the 1950's? that computes a gaussian process over the nearby weather stations with a prediction interval. If the neural networks never beat this, then whats the point of using all these frameworks? 

## Additional Spins/Questions 

Will it work with half the stations in Europe then tested on the other half?

## Timeline

**August 2nd: GitHub setup & literature review**

-Work on notebooks-

**August 8th: Input data pre-processed - Kriging algorithm experiment setup - Neural network experiment setup**

-In between we do checks-

**August 11th: Experiment run**

-Formulate research questions-

**August 13th: Writing begins**

-Write the damn thing-

**August 22nd: Writing is done**

-Paper Revision-

**August 28th: Submission**

