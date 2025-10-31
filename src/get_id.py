from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from datetime import datetime

API_KEY = ""


def get_youtube_service(api_key):
    return build('youtube', 'v3', developerKey=api_key)

def extract_channel_id(channel_url: str, api_key: str):
    try:
        youtube = get_youtube_service(api_key)

        if "channel/" in channel_url:
            # Direct channel ID provided
            return channel_url.split("channel/")[1].split("/")[0]

        elif "@" in channel_url:
            # Handle (new system)
            handle = channel_url.split("@")[1].split("/")[0]
            print(f"Searching for channel using handle: @{handle}")

            # First try to get channel directly using channels().list
            try:
                channel_response = youtube.channels().list(
                    part="id",
                    forHandle=handle
                ).execute()
                
                if channel_response["items"]:
                    return channel_response["items"][0]["id"]
            except HttpError:
                print("Direct handle lookup failed, trying search method...")

            # If direct lookup fails, use search with additional filters
            request = youtube.search().list(
                q=f"@{handle}",
                type="channel",
                part="snippet",
                maxResults=1
            )
            response = request.execute()

            if response["items"]:
                channel_id = response["items"][0]["snippet"]["channelId"]
                
                # Verify this is the correct channel by getting its details
                channel_details = youtube.channels().list(
                    part="snippet",
                    id=channel_id
                ).execute()
                
                if channel_details["items"]:
                    channel_handle = channel_details["items"][0]["snippet"].get("customUrl", "")
                    if channel_handle and handle.lower() in channel_handle.lower():
                        return channel_id
                    else:
                        print("Warning: Found channel handle doesn't match the requested handle")
                        return None
            else:
                print("Channel not found.")
                return None
        else:
            print("URL format not supported.")
            return None

    except HttpError as e:
        print(f"HTTP error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_uploads_playlist_id(channel_id):
    return "UU" + channel_id[2:]

def extract_playlist_id(playlist_url: str, api_key: str):
    """
    Extract playlist ID from a YouTube playlist URL.
    Handles both full URLs and direct playlist IDs.
    """
    try:
        youtube = get_youtube_service(api_key)

        if "playlist?list=" in playlist_url:
            # Extract playlist ID from URL
            return playlist_url.split("playlist?list=")[1].split("&")[0]
        elif "list=" in playlist_url:
            # Handle shortened URLs
            return playlist_url.split("list=")[1].split("&")[0]
        elif len(playlist_url) == 34 and playlist_url.startswith("PL"):
            # Direct playlist ID provided
            return playlist_url
        else:
            print("URL format not supported.")
            return None

    except HttpError as e:
        print(f"HTTP error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_videos_by_date_range(playlist_id, start_date, end_date, is_channel_uploads=False):
    """
    Get videos from a playlist or channel uploads playlist within a date range.
    
    Args:
        playlist_id (str): The playlist ID or channel uploads playlist ID
        start_date (datetime): Start date for filtering videos
        end_date (datetime): End date for filtering videos
        is_channel_uploads (bool): Whether this is a channel's uploads playlist
    
    Returns:
        list: List of video IDs that match the date criteria
    """
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    youtube = get_youtube_service(API_KEY)
    
    video_ids = []
    next_page_token = None
    total_videos_checked = 0
    
    print(f"\nSearching for videos between:")
    print(f"Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using playlist ID: {playlist_id}")
    
    try:
        # First, verify the playlist exists and get its details
        playlist_response = youtube.playlists().list(
            part="snippet,contentDetails",
            id=playlist_id
        ).execute()
        
        if not playlist_response["items"]:
            print("Error: Playlist not found!")
            return []
            
        playlist_title = playlist_response["items"][0]["snippet"]["title"]
        if is_channel_uploads:
            print(f"Channel uploads playlist found")
        else:
            print(f"Playlist found: {playlist_title}")
        
        while True:
            try:
                request = youtube.playlistItems().list(
                    part="contentDetails,snippet",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                
                response = request.execute()
                batch_videos = response["items"]
                total_videos_checked += len(batch_videos)
                
                # Process videos in batch
                for item in batch_videos:
                    video_id = item["contentDetails"]["videoId"]
                    published_at = item["snippet"]["publishedAt"]
                    published_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    
                    # Print video details for debugging
                    print(f"\nChecking video:")
                    print(f"Title: {item['snippet']['title']}")
                    print(f"Published at (UTC): {published_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Check if video is in range
                    is_in_range = start_date <= published_dt <= end_date
                    if is_in_range:
                        video_ids.append(video_id)
                        print(f"✅ Video is within date range")
                    else:
                        print(f"❌ Video is outside date range")
                        if published_dt < start_date:
                            print(f"   (Published before start date)")
                        else:
                            print(f"   (Published after end date)")
                
                # If we've found enough videos or reached the end, break
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
            except HttpError as e:
                print(f"HTTP error occurred: {e}")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                break
        
        print(f"\nTotal videos checked: {total_videos_checked}")
        print(f"Found {len(video_ids)} videos in the specified date range")
        return video_ids
        
    except Exception as e:
        print(f"Error accessing playlist: {e}")
        return []

def get_video_title(video_id, api_key):
    """
    Get the title of a YouTube video using its video ID.
    
    Args:
        video_id (str): The YouTube video ID
        api_key (str): YouTube Data API key
    
    Returns:
        str: The video title, or None if not found
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        
        if response["items"]:
            return response["items"][0]["snippet"]["title"]
        return None
    except Exception as e:
        print(f"Error getting video title: {e}")
        return None

# Example usage
if __name__ == "__main__":
    print("Choose input type:")
    print("1. Channel URL")
    print("2. Playlist URL")
    choice = input("Enter your choice (1 or 2): ")
    
    # Add time to the dates to ensure full day coverage
    start_date_str = input("Start date (YYYY-MM-DD): ")
    end_date_str = input("End date (YYYY-MM-DD): ")
    
    # Convert to datetime with time set to start and end of day
    # Note: We're using UTC time to match YouTube's API
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    
    print("\nDate range in UTC:")
    print(f"Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if choice == "1":
        channel_url = input("Enter the channel URL: ")
        channel_id = extract_channel_id(channel_url, API_KEY)
        if channel_id:
            print(f"\nChannel ID found: {channel_id}")
            playlist_id = get_uploads_playlist_id(channel_id)
            video_ids = get_videos_by_date_range(playlist_id, start_date, end_date, is_channel_uploads=True)
        else:
            print("Channel ID could not be retrieved.")
            video_ids = []
    else:
        playlist_url = input("Enter the playlist URL: ")
        playlist_id = extract_playlist_id(playlist_url, API_KEY)
        if playlist_id:
            print(f"\nPlaylist ID found: {playlist_id}")
            video_ids = get_videos_by_date_range(playlist_id, start_date, end_date)
        else:
            print("Playlist ID could not be retrieved.")
            video_ids = []
    
    if video_ids:
        print(f"\n🎯 Video IDs matching the date range:\n{video_ids}")
    else:
        print("\n❌ No videos found in the specified date range.")