from fire import Fire
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dataset.mine.parser import ChooseYourStoryParser

def main(quests_file: str, directory: str, timeout: int = 60*10):
    """
    Run ChooseYourStoryParser on a list of URLs.

    Parameters
    ----------
    quests_file : str
        Path to a file containing a newline-separated list of URLs to parse.
    directory : str
        Directory to save the parsed data.
    timeout : int, optional
        Timeout in seconds for each request. Default is 10 minutes.
    """
    quests = Path(quests_file).read_text().split('\n')
    for quest in quests:
        ChooseYourStoryParser(timeout).parse(quest, Path(directory))
        

if __name__ == '__main__':
    Fire(main)
