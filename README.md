# CaptaFlow (0.9.0)

CaptaFlow is a real-time sentiment analysis tool for identifying the dominant emotions in Twitch chat messages to provide thorough actionable audience engagement insights for creators.  

I've included an architecture diagram below to show the tools involved and the sequence they're used in. 

![Architecture Diagram](./files/Architecture-Diagram.png)


## Features

### Sentiment Analysis (with CardiffNLP) 

- The application passes batched Twitch chat messages through a RoBERTa transformer (CardiffNLP) to identify the dominant emotions in each batch.
- Each emotion receives a score from 0 to 1 representing how strongly that emotion is represented in that batch of messages.
- Each batch of emotions is given a datetime timestamp so that the exact moment in the twitch stream when each batch was taken can be pinpointed.
- Streamers can use the detected emotion levels as audience engagement metrics by connected them to the moment in stream when they were recorded. 


