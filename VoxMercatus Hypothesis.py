# ======| Imports | ======================================================
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import os
import re
from typing import Sequence, Dict, List, Union, Tuple, TypedDict
import datetime as dt
from tqdm import tqdm

# ======| Type Hint Classes | ======================================================
class _productsDescDict(TypedDict):
    SEARCH_QUERY: str
    RELEASE_DATE: str
    BLACKLIST: List[str]
    SEARCH_DAYS_BEFORE: int
    SEARCH_DAYS_AFTER: int

class _productDictionary(TypedDict):
    MAX_VID_RESULTS:int
    MAX_COMMENT_RESULTS: int
    PRODUCTS: Dict[str, _productsDescDict]

class _productSuccessDetails(TypedDict):
    model_sales: int
    model_price: int
    pred_sales: int
    pred_price: int
    model_budget: int
    average_top10_place_in_sales: Union[int, float]
    product_name: str

class _productSuccessMain(TypedDict):
    RTX_5090: Dict[str, _productSuccessDetails]

# ======| Main Hypothesis Proof Class | ======================================================
class PWQ_VM_AI_Hypothesis_Proof:
    CLS_ENTITY_PATTERNS: List[str] = [
        r'\bvs\.?\b',  # "vs" or "vs."
        r'\bversus\b',  # "versus"
        r'\b(vs|versus)\s+(the\s+)?',  # "vs the"
        r'\bthan\b',  # "better than", "worse than"
        r'\b(better|worse|faster|slower)\s+than\b',  # comparatives
        r'\b(kill|kills|killed)\s+(the\s+)?',  # "kills the iPhone"
        r'\b(beat|beats|beaten)\s+(the\s+)?',  # "beats the"
        r'\b(outperform|outperforms|outperformed)\s+(the\s+)?',
        r'\b(destroy|destroys|destroyed)\s+(the\s+)?',
        r'\b(crush|crushes|crushed)\s+(the\s+)?',
        r'\b(win|wins|won)\s+against\b',
        r'\b(lose|loses|lost)\s+to\b',
        r'\bcompared\s+to\b',
        r'\b(head[-\s]?to[-\s]?head)\b',  # head-to-head
        r'\bversus\b',  # versus
        r'\b(much|lot)\s+more\b',
        r'\b(pretty|ugly|bad|good|nice|terrible)\b'
    ]
    def __init__(self, product_dictionary:_productDictionary) -> None:
        """
        This is a class, which contains methods used to: prove PWQ-VM-AI hypothesis, scrape and filter youtube comments,
        and output .csv files per product with filtered, decisive and meaningful comments with opinion about the product.
        """
        DetectorFactory.seed = 42
        load_dotenv(r"C:\Users\pawli\OneDrive\Dokumenty\PWQ-VM-AI\API Data\Google Cloud API Data\.env")
        self.youtube_obj = build('youtube', 'v3', developerKey=os.getenv("API_KEY"))
        self.product_dictionary = product_dictionary
        self.max_vid_results: int = product_dictionary.get("MAX_VID_RESULTS")
        self.max_comment_results: int = product_dictionary.get("MAX_COMMENT_RESULTS")
        self.products: Dict[str, _productsDescDict] = product_dictionary.get("PRODUCTS")

    @staticmethod
    def _get_english(text, min_confidence=0.95):
        """
        Detect if text is English with confidence threshold
        """
        try:
            # Remove emojis and special chars that confuse detector
            clean_text = ' '.join(text.split())[:500]  # First 500 chars only
            lang = detect(clean_text)
            return lang == 'en'
        except LangDetectException:
            # If detection fails, check ASCII ratio as fallback
            ascii_chars = sum(ord(c) < 128 for c in text)
            ascii_ratio = ascii_chars / len(text) if text else 0
            return ascii_ratio > min_confidence  # 95% ASCII chars = probably English

    @staticmethod
    def Text_Cleaner(text:str) -> str:
        """Removes URLs, mentions, and extra noise."""
        text = str(text).replace('\n', ' ')  # Removes new lines
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Removes URLs
        text = re.sub(r'@\w+', '', text)  # Removes @ Tags
        text = " ".join(text.split())  # Removes Extra Spaces
        return text

    def Extract_Videos_From_Links(self, video_links: Union[str, List[str]]) -> List[Dict[str, str]]:
        """
        Accept YouTube video URLs and return video data
        """
        if isinstance(video_links, str):
            video_links = [video_links]

        video_data = []

        for link in video_links:
            # Extract video ID from URL
            video_id = None
            if 'youtu.be' in link:
                video_id = link.split('/')[-1].split('?')[0]
            elif 'youtube.com' in link:
                if 'v=' in link:
                    video_id = link.split('v=')[1].split('&')[0]
                elif 'embed/' in link:
                    video_id = link.split('embed/')[1].split('?')[0]

            if video_id:
                # Get video title using YouTube API
                try:
                    request = self.youtube_obj.videos().list(
                        part="snippet",
                        id=video_id
                    )
                    response = request.execute()

                    if response['items']:
                        title = response['items'][0]['snippet']['title']
                        video_data.append({
                            'id': video_id,
                            'title': title,
                            'url': link
                        })
                        print(f"✅ Added video: {title[:50]}...")
                    else:
                        print(f"⚠️ Could not fetch info for {link}")
                except Exception as e:
                    print(f"⚠️ Error fetching {link}: {e}")
            else:
                print(f"⚠️ Invalid YouTube URL: {link}")

        print(f"🎯 Loaded {len(video_data)} videos from provided links")
        return video_data

    def Extract_TopicQueried_PreLaunch_Videos(self, queries, blacklist_query: List[str],
                                              zulu_pblshd_before_date: str, zulu_pblshd_after_date: str,
                                              order: str = 'relevance', blacklist_prctng_removal: float = 0.125) -> \
            List[Dict[str, str]]:
        """Accept either a single query string or list of queries"""
        # Convert single string to list for uniform processing
        if isinstance(queries, str):
            queries = [queries]

        video_data: List = []
        seen_video_ids = set()

        for query in queries:
            print(f"🔍 Searching for: {query}")
            next_page_token = None
            max_api_results = 500

            while len(video_data) < self.max_vid_results:
                request = self.youtube_obj.search().list(
                    q=query,
                    part='snippet',
                    type='video',
                    maxResults=50,
                    publishedAfter=zulu_pblshd_after_date,
                    publishedBefore=zulu_pblshd_before_date,
                    order=order,
                    pageToken=next_page_token
                )
                res = request.execute()

                for item in res.get('items', []):
                    if 'videoId' not in item['id']:
                        continue
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']

                    if video_id not in seen_video_ids:
                        if not re.search("|".join(map(re.escape, blacklist_query)), title, re.IGNORECASE):
                            video_data.append({
                                'id': video_id,
                                'title': title
                            })
                            seen_video_ids.add(video_id)

                if len(video_data) >= self.max_vid_results:
                    break

                next_page_token = res.get('nextPageToken')
                if not next_page_token:
                    break

        query_type = "query" if len(queries) == 1 else "queries"
        print(f"🎯 Found {len(video_data)} videos across {len(queries)} {query_type}")
        return video_data[:self.max_vid_results]

    @staticmethod
    def Strip_Product_Model(product_name:str)->str:
        """Convert 'iPhone 17 Pro Max' → 'iphone' but handles multi-word brands"""
        name = str(product_name).lower()
        name = re.sub(r'\s*\d+.*$', '', name)
        suffixes = ['pro', 'plus', 'max', 'mini', 'ultra', 's', 'x', 'series', 'galaxy', 'fc', 'sports']
        words = name.split()
        result = []
        for word in words:
            if word in suffixes or len(result) >= 3:
                break  # Stop at first suffix
            result.append(word)
        if not result:
            result = [words[0]] if words else []

        return ' '.join(result).strip()

    def Comment_Extraction(self, video_id: str, txt_format: str = "plainText") -> List[str]:
        """Get ALL comments from a video with pagination"""
        comments: List = []
        next_page_token = None

        try:
            while len(comments) < self.max_comment_results:
                # Request up to 100 per page (max allowed)
                max_per_page = min(100, self.max_comment_results - len(comments))

                request = self.youtube_obj.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=max_per_page,
                    textFormat=txt_format,
                    pageToken=next_page_token
                )
                res = request.execute()

                # Extract comments
                page_comments = [
                    item['snippet']['topLevelComment']['snippet']['textDisplay']
                    for item in res.get('items', [])
                ]
                comments.extend(page_comments)

                # Check if we have enough or no more pages
                if len(comments) >= self.max_comment_results:
                    break

                next_page_token = res.get('nextPageToken')
                if not next_page_token:
                    break

        except Exception as e:
            print(f"⚠️ Skipping {video_id}: {str(e)[:100]}")

        return comments[:self.max_comment_results]

    def Calculate_End_And_Start_Search_Dates(self, search_days: str, timestamp: str, release_date:str) -> Tuple[str, str]:
        """ Calculates start and end dates from starting date and length of search in days."""
        release_date_dt = dt.datetime.strptime(release_date, "%d/%m/%Y")
        search_start_date = (release_date_dt).isoformat() + "Z"
        if timestamp == "past":
            search_end_date = (release_date_dt - dt.timedelta(days=search_days)).isoformat() + "Z"
            return search_start_date, search_end_date
        elif timestamp == "future":
            search_end_date = (release_date_dt + dt.timedelta(days=search_days)).isoformat() + "Z"
            return search_end_date, search_start_date

    def CSV_File_Construction(self, product_data: List, output_path: str, product_name: str, user_filename:str=None):
        """ Create a .csv file containing product data """
        product_data_filtered = [item for item in product_data if isinstance(item, dict)]
        spreadsheet = pd.DataFrame(product_data_filtered)
        product_folder = os.path.join(output_path, product_name.upper())
        os.makedirs(product_folder, exist_ok=True)
        if not spreadsheet.empty:
            start_count: int = spreadsheet.__len__()
            spreadsheet = spreadsheet.drop_duplicates(subset=['COMMENT'], keep='first')
            print(f"============================| {product_name} |=========================")
            print(f"\n✅ Analysis Complete!")
            print(f"Total processed: {start_count}")
            print(f"Final high-quality comments: {len(spreadsheet)}")
            filename: str = f"Product-{product_name}-CommentAnalysis.csv"
            if user_filename is not None:
                if user_filename.endswith('.csv'):
                    output_full_path = os.path.join(product_folder, user_filename)
                else:
                    user_filename += '.csv'
                    output_full_path = os.path.join(product_folder, user_filename)
            else:
                output_full_path = os.path.join(product_folder, filename)
            spreadsheet.to_csv(output_full_path, index=False)
        else:
            print("❌ No comments matched your criteria. Try a different search query.")

    def Data_Workflow_Construction(self, word_count_permission: int = 6) -> Dict[str, List]:
        """
        Automatically detects whether SEARCH_QUERY contains:
        - String → Query mode (YouTube search)
        - List of URLs → Links mode (direct video URLs)
        """
        all_products: Dict[str, List] = {}
        all_cls_entity_comments: List[List[Dict[str, str]]] = []

        print("\033[2J\033[H")  # Clear terminal
        print("🚀 YouTube Comment Scraper - Auto-Detect Mode")
        print("=" * 60 + "\n")

        # Create progress bars
        main_bar = tqdm(
            total=len(self.products),
            desc="📦 Overall Progress",
            position=0,
            bar_format="{l_bar}{bar:40}{r_bar}",
            leave=True
        )
        status_bar = tqdm(
            total=0,
            desc="📊 Current Status",
            position=1,
            bar_format="{desc}",
            leave=False
        )
        stats_bar = tqdm(
            total=0,
            desc="📈 Statistics: ",
            position=2,
            bar_format="{desc}",
            leave=False
        )

        for product_idx, (product, prod_data) in enumerate(self.products.items(), 1):
            main_bar.set_description(f"📦 Product {product_idx}/{len(self.products)}: {product[:20]}")
            main_bar.update(1)

            # Get SEARCH_QUERY – could be string or list
            search_input = prod_data.get("SEARCH_QUERY")

            # ===== AUTO-DETECT MODE =====
            if isinstance(search_input, str):
                # QUERY MODE – String means search YouTube
                mode = "query"
                status_bar.set_description(f"🔍 Searching videos for: {product}")

                blacklist: List[str] = prod_data.get("BLACKLIST", [])
                search_days: int = prod_data.get("SEARCH_DAYS")
                timestamp: int = prod_data.get("TIMESTAMP")
                release_date: str = prod_data.get("RELEASE_DATE")

                search_date_start, search_date_end = self.Calculate_End_And_Start_Search_Dates(
                    search_days, timestamp=timestamp, release_date=release_date
                )

                query_based_vids = self.Extract_TopicQueried_PreLaunch_Videos(
                    search_input, blacklist,
                    zulu_pblshd_after_date=search_date_end,
                    zulu_pblshd_before_date=search_date_start
                )

            elif isinstance(search_input, list):
                # LINKS MODE – List means direct YouTube URLs
                mode = "links"
                status_bar.set_description(f"🔍 Processing {len(search_input)} videos for: {product}")

                if not search_input:
                    print(f"\n⚠️ No video links provided for {product}, skipping...")
                    continue

                query_based_vids = self.Extract_Videos_From_Links(search_input)

            else:
                print(f"\n⚠️ Invalid SEARCH_QUERY for {product}, skipping...")
                continue

            # ===== COMMENT PROCESSING (same for both modes) =====
            status_bar.set_description(f"💬 Processing {len(query_based_vids)} videos")

            product_data: List = []
            cls_entity_comments: List = []
            total_comments_this_product = 0
            comparison_comments_this_product = 0

            video_bar = tqdm(
                total=len(query_based_vids),
                desc=f"📹 Videos for {product[:15]}",
                position=3,
                bar_format="{l_bar}{bar:30}{r_bar}",
                leave=False,
                unit="vid"
            )

            for vid_idx, vid in enumerate(query_based_vids, 1):
                video_id: str = vid['id']
                video_bar.set_description(f"📹 Video {vid_idx}/{len(query_based_vids)}")
                video_bar.update(1)

                vid_comments = self.Comment_Extraction(video_id)
                total_comments_this_product += len(vid_comments)

                stats_bar.set_description(
                    f"📈 Stats: {len(product_data)} comments | "
                    f"{comparison_comments_this_product} comparisons"
                )

                for comment in vid_comments:
                    clean_comment: str = self.Text_Cleaner(comment)
                    word_count: int = len(clean_comment.split())

                    if word_count >= word_count_permission:
                        try:
                            if self._get_english(clean_comment):
                                product_data.append({
                                    'video_title': vid['title'],
                                    'comment': clean_comment
                                })

                                CLS_ENTITY_REGEX = re.compile(
                                    '|'.join(
                                        f'({pattern})' for pattern in self.CLS_ENTITY_PATTERNS),
                                    re.IGNORECASE
                                )

                                stripped_product = self.Strip_Product_Model(str(product)).lower() if len(str(product).split()) >= 2 else str(product)
                                if "RTX" in product or "5090" in product:
                                    discussion_words = [" 5090 ", " rig ", " 1440p ", " 4K ", " 1080p ", "GB ", " vram ", " price ", stripped_product]
                                elif "iphone" in product.lower():
                                    discussion_words = [" battery ", " device ", " style ", " iphone ", " design ", " price ",stripped_product]
                                elif "avatar" in product.lower():
                                    discussion_words = [" quarich "," quaritch "," na'vi ", " navi ",stripped_product]
                                words_in_comment = [word in clean_comment.lower() for word in discussion_words]
                                has_keyword = any(word in clean_comment.lower() for word in discussion_words)
                                has_comparison = CLS_ENTITY_REGEX.search(clean_comment) is not None
                                if has_comparison and has_keyword:
                                    matched_word = next((word for word in discussion_words
                                                         if word in clean_comment.lower()), None)
                                    cls_entity_comments.append({
                                        'PRODUCT':str(product),
                                        'TARGET': str(matched_word).lower(),
                                        "COMMENT": clean_comment,
                                        "VIDEO_TITLE": vid['title']
                                    })
                                    comparison_comments_this_product += 1
                        except Exception as e:
                            continue

            video_bar.close()

            # Save to CSV
            self.CSV_File_Construction(
                cls_entity_comments,
                r"C:\Users\pawli\OneDrive\Dokumenty\PWQ-VM-AI\Hypothesis Comment Files",
                product.__str__(),
                user_filename=product.__str__()+"V3"
            )

            all_products[f"{product}"] = product_data
            all_cls_entity_comments.append(cls_entity_comments)

            status_bar.set_description(
                f"✅ {product}: {len(product_data)} comments, {len(cls_entity_comments)} comparisons")
            stats_bar.set_description(
                f"📈 Cumulative: {sum(len(p) for p in all_products.values())} total | "
                f"{sum(len(p) for p in all_cls_entity_comments)} comparisons"
            )

        # Close all bars
        status_bar.close()
        stats_bar.close()
        main_bar.close()

        print("\033[2K\033[1A" * 4)
        print("\n" + "=" * 60)
        print("🎉 DATA COLLECTION COMPLETE!")
        print("=" * 60)

        for product_comments in all_cls_entity_comments:
            if product_comments:
                product_name = product_comments[0].get('PRODUCT_NAME', 'Unknown')
                print(f"\n📝 {product_name}: {len(product_comments)} comparison comments")

        return all_products
