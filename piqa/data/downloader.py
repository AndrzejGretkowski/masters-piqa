import requests
from requests.compat import urljoin
from pathlib import Path
from piqa.data.constants import PIQA_DATA_URL, PIQA_DATA_SETS, LOCAL_FILE_LOCATION


class Downloader(object):
    def __init__(self, download=False):
        self._download = download

    @staticmethod
    def _download_file(file_url, local_filename):
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def download_set(self, set_name):
        if set_name not in PIQA_DATA_SETS:
            return False

        if self._verify_files(set_name):
            return True

        for set_file in PIQA_DATA_SETS[set_name]:
            if set_file is not None:
                local_name = Path(LOCAL_FILE_LOCATION, set_name, set_file)
                self._download_file(
                    urljoin(PIQA_DATA_URL, set_file),
                    local_name
                )

        return True

    def download(self):
        if self.is_download_needed():
            for set_name in PIQA_DATA_SETS.keys():
                _ = self.download_set(set_name)

    def is_download_needed(self):
        return any([not self._verify_files(set_name) for set_name in PIQA_DATA_SETS.keys()])

    def _verify_files(self, set_name):
        if set_name not in PIQA_DATA_SETS:
            return True
        for set_file in PIQA_DATA_SETS[set_name]:
            if set_file is not None:
                if not Path(LOCAL_FILE_LOCATION, set_name, set_file).exists():
                    return False
        return True
