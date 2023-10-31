import string
from typing import List, Tuple


class AccentProcessor:
    def __init__(self) -> None:
        self.__map_word_accent = {
            "ngang": [
                "a",
                "ă",
                "â",
                "e",
                "ê",
                "i",
                "o",
                "ô",
                "ơ",
                "u",
                "ư",
                "y",
                "A",
                "Ă",
                "Â",
                "E",
                "Ê",
                "I",
                "O",
                "Ô",
                "Ơ",
                "U",
                "Ư",
                "Y",
            ],
            "sac": [
                "á",
                "ắ",
                "ấ",
                "é",
                "ế",
                "í",
                "ó",
                "ố",
                "ớ",
                "ú",
                "ứ",
                "ý",
                "Á",
                "Ắ",
                "Ấ",
                "É",
                "Ế",
                "Í",
                "Ó",
                "Ố",
                "Ớ",
                "Ú",
                "Ứ",
                "Ý",
            ],
            "huyen": [
                "à",
                "ằ",
                "ầ",
                "è",
                "ề",
                "ì",
                "ò",
                "ồ",
                "ờ",
                "ù",
                "ừ",
                "ỳ",
                "À",
                "Ằ",
                "Ầ",
                "È",
                "Ề",
                "Ì",
                "Ò",
                "Ồ",
                "Ờ",
                "Ù",
                "Ừ",
                "Ỳ",
            ],
            "hoi": [
                "ả",
                "ẳ",
                "ẩ",
                "ẻ",
                "ể",
                "ỉ",
                "ỏ",
                "ổ",
                "ở",
                "ủ",
                "ử",
                "ỷ",
                "Ả",
                "Ẳ",
                "Ẩ",
                "Ẻ",
                "Ể",
                "Ỉ",
                "Ỏ",
                "Ổ",
                "Ở",
                "Ủ",
                "Ử",
                "Ỷ",
            ],
            "nga": [
                "ã",
                "ẵ",
                "ẫ",
                "ẽ",
                "ễ",
                "ĩ",
                "õ",
                "ỗ",
                "ỡ",
                "ũ",
                "ữ",
                "ỹ",
                "Ã",
                "Ẵ",
                "Ẫ",
                "Ẽ",
                "Ễ",
                "Ĩ",
                "Õ",
                "Ỗ",
                "Ỡ",
                "Ũ",
                "Ữ",
                "Ỹ",
            ],
            "nang": [
                "ạ",
                "ặ",
                "ậ",
                "ẹ",
                "ệ",
                "ị",
                "ọ",
                "ộ",
                "ợ",
                "ụ",
                "ự",
                "ỵ",
                "Ạ",
                "Ặ",
                "Ậ",
                "Ẹ",
                "Ệ",
                "Ị",
                "Ọ",
                "Ộ",
                "Ợ",
                "Ụ",
                "Ự",
                "Ỵ",
            ],
        }
        self.__char_i = "i ì í ỉ ĩ ị I Ì Í Ỉ Ĩ Ị"
        self.__char_u = "u ù ú ủ ũ ụ U Ù Ú Ủ Ũ Ụ"
        self.__char_vi = (
            "ă ằ ắ ẳ ẵ ặ â ầ ấ ẩ ẫ ậ ê ề ế ể ễ ệ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ ư ừ ứ ử ữ ự Ă Ằ Ắ Ẳ Ẵ Ặ Â Ầ Ấ Ẩ"
            " Ẫ Ậ Ê Ề Ế Ể Ễ Ệ Ô Ồ Ố Ổ Ỗ Ộ Ơ Ờ Ớ Ở Ỡ Ợ Ư Ừ Ứ Ử Ữ Ự"
        )
        self.__initialize()

    def __initialize(self):
        self.__list_vowel = []
        self.__characters = string.ascii_letters
        for key in self.__map_word_accent:
            self.__list_vowel.extend(self.__map_word_accent[key])
            self.__characters += "".join(self.__map_word_accent[key])
            self.__characters += "đĐ"

    def __split_word(self, word: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Split word into previous consonant, vowel, following consonant
        Args:
            word (str): Input word.

        Returns:
            Tuple[List[str], List[str], List[str]]: Three list of previous consonant, vowel, following consonant
        """
        before = []
        vowel = []
        after = []
        flag = True
        word_list = list(word)
        for i in range(len(word_list)):
            if word_list[i] in self.__list_vowel and len(after) == 0:
                flag = False
                vowel.append(word_list[i])
            elif flag:
                before.append(word_list[i])
            else:
                after.append(word_list[i])
        return before, vowel, after

    def __detect_accent(self, word: str) -> str:
        """
        Detect accent of input word.
        Args:
            word (str): Input word.

        Returns:
            str: Accent of input word.
        """
        for key in self.__map_word_accent.keys():
            if key != "ngang":
                for char in self.__map_word_accent[key]:
                    if char in word:
                        return key
        return "ngang"

    def __change_accent(self, char: str, to_accent: str) -> str:
        """
        Change accent of a character.
        Args:
            char (str): Input character.
            to_accent (str): Input accent to change character to.

        Returns:
            (str): Accent changed character.
        """
        index = None
        for key in self.__map_word_accent.keys():
            if char in self.__map_word_accent[key]:
                index = self.__map_word_accent[key].index(char)
                break
        return self.__map_word_accent[to_accent][index]

    def __preprocess_word(self, word: str) -> Tuple[str, str, str]:
        """
        Preprocess word before accent processing.
        Args:
            word (str): Input word.

        Returns:
            Tuple[str, str, str]: Tuple of preprocessed word.
        """
        begin = ""
        end = ""
        while word[0] not in self.__characters:
            begin += word[0]
            word = word[1:]
        while word[-1] not in self.__characters:
            end = word[-1] + end
            word = word[:-1]
        return begin, word, end

    def process_accent_by_word(self, word: str) -> str:
        """
        Normalize accent of input word.
        Args:
            word (str): Input word.

        Returns:
            (str): Accent normalized word.
        """
        if word.strip() == "":
            return word
        begin, word, end = self.__preprocess_word(word)
        before, vowel, after = self.__split_word(word)
        if len(vowel) > 0:
            word_accent = self.__detect_accent(vowel)
            if self.__detect_accent == "ngang":
                return begin + word + end
            check_vi = False
            index_vi = -1
            count_vi = 0
            for char in vowel:
                if char in self.__char_vi:
                    index_vi = vowel.index(char)
                    count_vi += 1
                    check_vi = True
            if check_vi and (count_vi == 1):
                for i in range(len(vowel)):
                    if i != index_vi:
                        vowel[i] = self.__change_accent(vowel[i], "ngang")
                    else:
                        vowel[i] = self.__change_accent(vowel[i], word_accent)
            elif len(vowel) == 3:
                vowel[0] = self.__change_accent(vowel[0], "ngang")
                if len(after) != 0:
                    vowel[1] = self.__change_accent(vowel[1], "ngang")
                    vowel[2] = self.__change_accent(vowel[2], word_accent)
                else:
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                    vowel[2] = self.__change_accent(vowel[2], "ngang")
            elif len(vowel) == 2:
                if len(after) != 0:
                    vowel[0] = self.__change_accent(vowel[0], "ngang")
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                elif (
                    len(before) == 1
                    and (before[0] == "q")
                    and (vowel[0] in self.__char_u)
                ):
                    vowel[0] = self.__change_accent(vowel[0], "ngang")
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                elif (
                    len(before) == 1
                    and (before[0] == "g")
                    and (vowel[0] in self.__char_i)
                ):
                    vowel[0] = self.__change_accent(vowel[0], "ngang")
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                else:
                    vowel[0] = self.__change_accent(vowel[0], word_accent)
                    vowel[1] = self.__change_accent(vowel[1], "ngang")
            else:
                vowel[0] = self.__change_accent(vowel[0], word_accent)

        return begin + "".join(before + vowel + after) + end

    def process_accent_by_document(self, document: str) -> str:
        """
        Normalize accent words in document.
        Args:
            document (str): Input document.

        Returns:
            str: Normalized accent words in document.
        """
        processed = []
        word = ""
        for char in document:
            if char in self.__characters:
                word += char
            else:
                processed.append(self.process_accent_by_word(word) + char)
                word = ""
        if word != "":
            processed.append(word)
        return "".join(processed).strip()

    def custom_change_accent(self, word: str, word_accent: str) -> str:
        """
        Change accent of word to input accent.
        Args:
            word (str): Input word.
            word_accent (str): Input accent to change word to. Just in ["sac", "huyen", "hoi", "nga", "nang", "ngang"]

        Returns:
            (str): Accent changed word.
        """
        if word_accent in self.__map_word_accent.keys():
            before, vowel, after = self.__split_word(word)
            check_vi = False
            index_vi = -1
            count_vi = 0
            for char in vowel:
                if char in self.__char_vi:
                    index_vi = vowel.index(char)
                    count_vi += 1
                    check_vi = True
            if check_vi and (count_vi == 1):
                for i in range(len(vowel)):
                    if i != index_vi:
                        vowel[i] = self.__change_accent(vowel[i], "ngang")
                    else:
                        vowel[i] = self.__change_accent(vowel[i], word_accent)
            elif len(vowel) == 3:
                vowel[0] = self.__change_accent(vowel[0], "ngang")
                if len(after) != 0:
                    vowel[1] = self.__change_accent(vowel[1], "ngang")
                    vowel[2] = self.__change_accent(vowel[2], word_accent)
                else:
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                    vowel[2] = self.__change_accent(vowel[2], "ngang")
            elif len(vowel) == 2:
                if len(after) != 0:
                    vowel[0] = self.__change_accent(vowel[0], "ngang")
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                elif (
                    len(before) == 1
                    and (before[0] == "q")
                    and (vowel[0] in self.__char_u)
                ):
                    vowel[0] = self.__change_accent(vowel[0], "ngang")
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                elif (
                    len(before) == 1
                    and (before[0] == "g")
                    and (vowel[0] in self.__char_i)
                ):
                    vowel[0] = self.__change_accent(vowel[0], "ngang")
                    vowel[1] = self.__change_accent(vowel[1], word_accent)
                else:
                    vowel[0] = self.__change_accent(vowel[0], word_accent)
                    vowel[1] = self.__change_accent(vowel[1], "ngang")
            else:
                vowel[0] = self.__change_accent(vowel[0], word_accent)

            return "".join(before + vowel + after)
        else:
            return word
