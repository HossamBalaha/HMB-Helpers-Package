class StringsHelper(object):
  r'''
  StringsHelper: Helper methods for common string operations.

  Initialize with a string and call instance methods to perform queries and
  small algorithms (permutations, compression, palindrome-permutation checks,
  etc.). All public methods are designed to be small, self-contained, and
  documented inline.
  '''

  def __init__(self, string):
    r'''
    Constructor for the StringsHelper class.

    Parameters:
     string (str): The string to be used by the class.
    '''

    self.__string = string

  def GetString(self):
    r'''
    Returns the string that was set during creating the object.

    Returns:
      str: The string that was set during creating the object.
    '''

    return self.__string

  def GetStringLength(self):
    r'''
    Returns the length of the string that was set during creating the object.

    Returns:
      int: The length of the string that was set during creating the object.
    '''

    return len(self.__string)

  def SetString(self, string):
    r'''
    Sets the string to a new value.

    Parameters:
      string (str): The new value for the string.
    '''

    self.__string = string

  def GetCharAt(self, index):
    r'''
    Returns the character at the specified index.

    Parameters:
      index (int): The index of the character to be returned.

    Returns:
      str: The character at the specified index.
    '''

    if (index < 0 and -index >= len(self.__string)):
      return self.__string[0]

    if (index >= len(self.__string)):
      return self.__string[-1]

    return self.__string[index]

  def GetCharIndex(self, char):
    r'''
    Returns the index of the specified character.

    Parameters:
      char (str): The character to be searched for.

    Returns:
      int: The index of the specified character.
    '''

    return self.__string.index(char)

  def GetCharCount(self, char):
    r'''
    Returns the number of times the specified character appears in the string.

    Parameters:
      char (str): The character to be searched for.

    Returns:
      int: The number of times the specified character appears in the string.
    '''

    # Empty token has undefined count in Python (returns len+1); treat as 0 for tests.
    if (char is None or len(char) == 0):
      return 0
    return self.__string.count(char)

  def GetCharCountFrom(self, char, index):
    r'''
    Returns the number of times the specified character appears in the string

    Parameters:
      char (str): The character to be searched for.
      index (int): The index from which the search will start.

    Returns:
      int: The number of times the specified character appears in the string.
    '''

    if (char is None or len(char) == 0):
      return 0
    start = max(0, int(index))
    return self.__string[start:].count(char)

  def GetCharCountTo(self, char, index):
    r'''
    Returns the number of times the specified character appears in the string

    Parameters:
      char (str): The character to be searched for.
      index (int): The index to which the search will end.

    Returns:
      int: The number of times the specified character appears in the string.
    '''

    if (char is None or len(char) == 0):
      return 0
    end = min(len(self.__string), max(0, int(index)))
    return self.__string[:end].count(char)

  def GetCharCountBetween(self, char, index1, index2):
    r'''
    Returns the number of times the specified character appears in the string

    Parameters:
      char (str): The character to be searched for.
      index1 (int): The index from which the search will start.
      index2 (int): The index to which the search will end.

    Returns:
      int: The number of times the specified character appears in the string.
    '''

    if (char is None or len(char) == 0):
      return 0
    i1 = max(0, int(index1))
    i2 = min(len(self.__string), max(0, int(index2)))
    if (i2 < i1):
      return 0
    return self.__string[i1:i2].count(char)

  def GetCharCountBetweenInclusive(self, char, index1, index2):
    r'''
    Returns the number of times the specified character appears in the string

    Parameters:
      char (str): The character to be searched for.
      index1 (int): The index from which the search will start.
      index2 (int): The index to which the search will end.

    Returns:
      int: The number of times the specified character appears in the string.
    '''

    if (char is None or len(char) == 0):
      return 0
    i1 = max(0, int(index1))
    i2 = min(len(self.__string) - 1, max(0, int(index2)))
    if (i2 < i1):
      return 0
    return self.__string[i1:i2 + 1].count(char)

  def GetCharCountBetweenExclusive(self, char, index1, index2):
    r'''
    Returns the number of times the specified character appears in the string

    Parameters:
      char (str): The character to be searched for.
      index1 (int): The index from which the search will start.
      index2 (int): The index to which the search will end.

    Returns:
      int: The number of times the specified character appears in the string.
    '''

    if (char is None or len(char) == 0):
      return 0
    i1 = max(0, int(index1) + 1)
    i2 = min(len(self.__string), max(0, int(index2)))
    if (i2 <= i1):
      return 0
    return self.__string[i1:i2].count(char)

  def GetReverse(self):
    r'''
    Returns the reverse of the string.

    Returns:
      str: The reverse of the string.
    '''

    return self.__string[::-1]

  def GetReverseFrom(self, index):
    r'''
    Returns the reverse of the string from the specified index.

    Parameters:
      index (int): The index from which the reverse will start.

    Returns:
      str: The reverse of the string from the specified index.
    '''

    return self.__string[index:][::-1]

  def GetReverseTo(self, index):
    r'''
    Returns the reverse of the string to the specified index.

    Parameters:
      index (int): The index to which the reverse will end.

    Returns:
      str: The reverse of the string to the specified index.
    '''

    return self.__string[:index][::-1]

  def GetReverseBetween(self, index1, index2):
    r'''
    Returns the reverse of the string between the specified indexes.

    Parameters:
      index1 (int): The index from which the reverse will start.
      index2 (int): The index to which the reverse will end.

    Returns:
      str: The reverse of the string between the specified indexes.
    '''

    return self.__string[index1:index2][::-1]

  def GetReverseBetweenInclusive(self, index1, index2):
    r'''
    Returns the reverse of the string between the specified indexes.

    Parameters:
      index1 (int): The index from which the reverse will start.
      index2 (int): The index to which the reverse will end.

    Returns:
      str: The reverse of the string between the specified indexes.
    '''

    return self.__string[index1:index2 + 1][::-1]

  def GetReverseBetweenExclusive(self, index1, index2):
    r'''
    Returns the reverse of the string between the specified indexes.

    Parameters:
      index1 (int): The index from which the reverse will start.
      index2 (int): The index to which the reverse will end.

    Returns:
      str: The reverse of the string between the specified indexes.
    '''

    return self.__string[index1 + 1:index2][::-1]

  def IsSubStringFrom(self, string):
    r'''
    Returns True if the string is a substring of the string that was set during creating the object.

    Parameters:
      string (str): The string to be searched for.

    Returns:
      bool: True if the string is a substring of the string that was set during creating the object.
    '''

    return self.__string in string

  def IsSubStringTo(self, string):
    r'''
    Returns True if the string is a substring of the string that was set during creating the object.

    Parameters:
      string (str): The string to be searched for.

    Returns:
      bool: True if the string is a substring of the string that was set during creating the object.
    '''

    return string in self.__string

  def IsRotationWith(self, string):
    r'''
    Returns True if the string is a rotation of the string that was set during creating the object.

    Parameters:
      string (str): The string to be searched for.

    Returns:
      bool: True if the string is a rotation of the string that was set during creating the object.
    '''

    if (self.GetStringLength() <= 0 or len(string) <= 0):
      return False
    if (self.GetStringLength() != len(string)):
      return False
    return self.IsSubStringFrom(string + self.__string)

  def GetPermutations(self):
    r'''
    Returns a list of all the permutations of the string that was set during creating the object.

    Returns:
      list: A list of all the permutations of the string that was set during creating the object.
    '''

    # For performance and correctness, return unique sorted permutations generated by itertools
    import itertools
    perms = set(''.join(p) for p in itertools.permutations(self.__string))
    return sorted(perms)

  def IsPermutationOf(self, string):
    r'''
    Returns True if the string is a permutation of the string that was set during creating the object.

    Parameters:
      string (str): The string to be searched for.

    Returns:
      bool: True if the string is a permutation of the string that was set during creating the object.
    '''

    if (not isinstance(string, str)):
      return False
    if (len(string) != self.GetStringLength()):
      return False
    # Simple and robust permutation check: compare sorted characters
    return sorted(string) == sorted(self.__string)

  def IsPalindromePermutation(self):
    r'''
    Returns True if the string is a palindrome permutation of the string that was set during creating the object.

    Returns:
      bool: True if the string is a palindrome permutation of the string that was set during creating the object.
    '''

    import string

    letters = string.ascii_letters
    dictLetters = {}
    for letter in self.__string:
      if (letters in letters):
        dictLetters.setdefault(letter, 0)
        dictLetters[letter] += 1
    countOdd = 0
    for key in dictLetters.keys():
      if (dictLetters[key] % 2 != 0):
        countOdd += 1
    return (countOdd <= 1)

  def IsUniqueCharacters(self):
    r'''
      Returns True if the string has all unique characters.

    Returns:
      bool: True if the string has all unique characters.
    '''

    return self.GetStringLength() == len(set(self.__string))

  def Urlify(self, strip=True):
    r'''
    Returns the string with all spaces replaced by '%20'.

    Parameters:
      strip (bool): If True, the string will be stripped before replacing the spaces.

    Returns:
      str: The string with all spaces replaced by '%20'.
    '''

    newString = self.__string[:]

    if (strip):
      newString = newString.strip()

    newString = newString.replace(" ", "%20")
    return newString

  def Compress(self):
    r'''
    Returns the string compressed.

    Returns:
      str: The string compressed.
    '''

    seenLetters = []
    compressedString = ""
    for letter in self.__string:
      if (letter not in seenLetters):
        seenLetters.append(letter)
        compressedString += letter + str(self.__string.count(letter))
    isBetter = len(compressedString) < self.GetStringLength()
    return (compressedString, isBetter)

  def IsOneEditOf(self, string):
    r'''
    Returns True if the string is one edit of the string that was set during creating the object.

    Parameters:
      string (str): The string to be searched for.

    Returns:
      bool: True if the string is one edit of the string that was set during creating the object.
    '''

    if (string == self.__string):
      return False
    outLen = len(string)
    absLenDiff = abs(outLen - self.GetStringLength())
    if (absLenDiff > 1):
      return False
    maxLen = max(outLen, self.GetStringLength())
    for i in range(maxLen):
      if (i < outLen and i < self.GetStringLength()):
        eA = self.__string[i]
        eB = string[i]
        if (eA == eB):
          continue
        if (outLen < self.GetStringLength()):
          newA = self.__string.replace(eA, "")
          if (newA == string):
            return True
        elif (outLen > self.GetStringLength()):
          newB = string.replace(eB, "")
          if (self.__string == newB):
            return True
        else:
          newA = self.__string.replace(eA, "")
          newB = string.replace(eB, "")
          if (newA == newB):
            return True
      else:
        if (outLen < self.GetStringLength()):
          newA = self.__string.replace(self.__string[-1], "")
          if (newA == string):
            return True
        else:
          newB = string.replace(string[-1], "")
          if (self.__string == newB):
            return True

    return False


# Add a lightweight demo to exercise each method safely.
if __name__ == "__main__":
  # SafeCall helper used to call methods and gracefully report failures.
  def SafeCall(name, fn, *args, **kwargs):
    try:
      res = fn(*args, **kwargs)
      print(f"{name} ->", res)
      print("-" * 40)
      return res
    except Exception as e:
      print(f"{name} raised {type(e).__name__}:", e)
      print("-" * 40)
      return None


  # Sample usage covering typical edge cases.
  s = "aabccde"
  sh = StringsHelper(s)

  SafeCall("GetString", sh.GetString)
  SafeCall("GetStringLength", sh.GetStringLength)
  SafeCall("GetCharAt 0", sh.GetCharAt, 0)
  SafeCall("GetCharAt -1", sh.GetCharAt, -1)
  SafeCall("GetCharAt out-of-range", sh.GetCharAt, 100)

  SafeCall("GetCharIndex (existing)", sh.GetCharIndex, "a")
  SafeCall("GetCharIndex (missing)", sh.GetCharIndex, "z")

  SafeCall("GetCharCount 'a'", sh.GetCharCount, "a")
  SafeCall("GetCharCountFrom 'c' from 2", sh.GetCharCountFrom, "c", 2)
  SafeCall("GetCharCountTo 'c' to 4", sh.GetCharCountTo, "c", 4)
  SafeCall("GetCharCountBetween 'b' 1-5", sh.GetCharCountBetween, "b", 1, 5)
  SafeCall("GetCharCountBetweenInclusive 'b' 1-5 incl", sh.GetCharCountBetweenInclusive, "b", 1, 5)
  SafeCall("GetCharCountBetweenExclusive 'b' 1-5 excl", sh.GetCharCountBetweenExclusive, "b", 1, 5)

  SafeCall("GetReverse", sh.GetReverse)
  SafeCall("GetReverseFrom 2", sh.GetReverseFrom, 2)
  SafeCall("GetReverseTo 4", sh.GetReverseTo, 4)
  SafeCall("GetReverseBetween 1-5", sh.GetReverseBetween, 1, 5)
  SafeCall("GetReverseBetweenInclusive 1-5", sh.GetReverseBetweenInclusive, 1, 5)
  SafeCall("GetReverseBetweenExclusive 1-5", sh.GetReverseBetweenExclusive, 1, 5)

  SafeCall("IsSubStringFrom (check stored in other)", sh.IsSubStringFrom, "dummy")
  SafeCall("IsSubStringTo (check other in stored)", sh.IsSubStringTo, "abc")

  SafeCall("IsRotationWith (same length, not rotated)", sh.IsRotationWith, "aabccde")
  SafeCall("IsRotationWith (rotated)", sh.IsRotationWith, "de aabcc".replace(" ", ""))

  SafeCall("GetPermutations", sh.GetPermutations)
  SafeCall("IsPermutationOf (same)", sh.IsPermutationOf, "aabccde")
  SafeCall("IsPalindromePermutation", sh.IsPalindromePermutation)
  SafeCall("IsUniqueCharacters", sh.IsUniqueCharacters)

  SafeCall("Urlify (strip)", sh.Urlify, True)
  SafeCall("Compress", sh.Compress)

  SafeCall("IsOneEditOf (one edit)", sh.IsOneEditOf, "aabccde")
  SafeCall("IsOneEditOf (different)", sh.IsOneEditOf, "aabccdefg")

  print("StringsHelper demo completed.")
