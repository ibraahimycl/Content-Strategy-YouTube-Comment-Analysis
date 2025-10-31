from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from datetime import datetime
import csv

API_KEY = ""

def video_comments(video_id, output_filename=None):
    """
    Get comments for a video and save them to a CSV file.
    
    Args:
        video_id (str): The YouTube video ID
        output_filename (str, optional): The name of the output CSV file. If not provided, 
                                       a default name with timestamp will be used.
    
    Returns:
        str: The name of the created CSV file
    """
    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    # Create CSV file
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"comments_{video_id}_{timestamp}.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Create CSV writer
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Comment', 'Published At', 'Like Count', 'Reply Count'])
        
        video_response = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100  # Maximum allowed by API
        ).execute()

        while video_response:
            for item in video_response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comment_text = comment["textDisplay"]
                published_at = comment["publishedAt"]
                like_count = comment["likeCount"]
                reply_count = item["snippet"]["totalReplyCount"]
                
                # Write main comment to CSV
                csv_writer.writerow([comment_text, published_at, like_count, reply_count])
                
                # Get replies if any
                if reply_count > 0:
                    for reply in item["replies"]["comments"]:
                        reply_snippet = reply["snippet"]
                        reply_text = reply_snippet["textDisplay"]
                        reply_published_at = reply_snippet["publishedAt"]
                        reply_like_count = reply_snippet["likeCount"]
                        
                        # Write reply to CSV
                        csv_writer.writerow([f"Reply: {reply_text}", reply_published_at, reply_like_count, 0])

            # Check if there are more pages of comments
            if "nextPageToken" in video_response:
                video_response = youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    pageToken=video_response["nextPageToken"],
                    maxResults=100
                ).execute()
            else:
                break
    
    return output_filename

if __name__ == "__main__":
    # This part is only for testing the script directly
    video_id = input("Please video link: ")
    if 'youtube.com' in video_id or 'youtu.be' in video_id:
        # Extract video ID from URL
        if "watch?v=" in video_id:
            video_id = video_id.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in video_id:
            video_id = video_id.split("youtu.be/")[1].split("?")[0]
    
    output_file = video_comments(video_id)
    print(f"\nComments have been saved to {output_file}")