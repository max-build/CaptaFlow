# CaptaFlow (0.9.1)

CaptaFlow is a real-time sentiment data analysis application for identifying the dominant emotions in Twitch chat messages to provide audience engagement insights for creators.  


## Architecture Diagram 
I've included an architecture diagram below to show the tools involved and the sequence they're used in. 

![Architecture Diagram](./files/Architecture-Diagram.png)



## Features

### Sentiment Analysis (with CardiffNLP) 

- The application passes batched Twitch chat messages through a RoBERTa transformer (CardiffNLP) to identify the dominant emotions in each batch.
- Each emotion receives a score from 0 to 1 representing how strongly that emotion is represented in that batch of messages.
- Each batch of emotions is given a datetime timestamp so that the exact moment in the twitch stream when each batch was taken can be pinpointed.
- Streamers can use the detected emotion levels as audience engagement metrics by connecting them to the moment in stream when they were recorded. 



## NLP Pipeline Output Previews
**These are displayed in terminal as the application runs and is seperate from visualisations as shown in QuickSuite.**

### Message Readings 

![Message Preview](./files/Message-Results.png)

This console log displays the readings that are taken for each message.
  - Note, TRUST, DISGUST, ANTICIPATION, LOVE and PESSIMISM are not included in the final batch result previews or the final reports, the reason for this is I concluded that they didn't provide any additional insight into the
    dominant moods in chat and overlapped too heavily with the other dominant emotions and so they were redundant. 

### Batch Results

![Batch Results](./files/Batch-Results.png) 

This console log displays the total sentiment scores for each batch of messages (10 messages per batch by default). 

### Readings Summary 

![Readings-Summary](./files/Readings-Summary.png)

This console log prints all rows from the dataframe which all batch readings are added to, along with timestamps for when they were taken, and the stream ID the readings belong to. 



# Patch Notes

## 0.9.1 (24/01/26)
- Revised method of exporting batch insight scores to S3.
- Batch scores are now stored in a list of tuples (instead of being appended to a dataframe as before) prior to exporting to S3.
- Appending batch scores to the end of the dataframe had O(n) timing vs appending to a list of tuples which is O(1).
- This allows the application to be more scalable and handle a higher velocity of concurrent throughput. 

