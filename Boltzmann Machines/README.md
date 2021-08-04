## Boltzmann Machines + Stacked Autoencoders  

From Amazon product suggestions to Netflix movie recommendations - good recommender systems are very valuable in today's World. And specialists who can create them are some of the top-paid Data Scientists on the planet.  

We will work on a dataset that has exactly the same features as the Netflix dataset: plenty of movies, thousands of users, who have rated the movies they watched. The ratings go from 1 to 5, exactly like in the Netflix dataset, which makes the Recommender System more complex to build than if the ratings were simply “Liked” or “Not Liked”.   

Your final Recommender System will be able to predict the ratings of the movies the customers didn’t watch. Accordingly, by ranking the predictions from 5 down to 1, your Deep Learning model will be able to recommend which movies each user should watch. Creating such a powerful Recommender System is quite a challenge so we will give ourselves two shots. Meaning we will build it with two different Deep Learning models.  

Our first model will be Deep Belief Networks, complex Boltzmann Machines that will be covered in Part 5. Then our second model will be with the powerful AutoEncoders, my personal favorites. You will appreciate the contrast between their simplicity, and what they are capable of.  

And you will even be able to apply it to yourself or your friends. The list of movies will be explicit so you will simply need to rate the movies you already watched, input your ratings in the dataset, execute your model and voila! The Recommender System will tell you exactly which movies you would love one night you if are out of ideas of what to watch on Netflix!  
