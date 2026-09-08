import math
import re
import unicodedata
from collections import Counter, defaultdict

modeThresholds = {"conservative": 0.68, "balanced": 0.55, "exploratory": 0.45}


def tokenize(text):
    return re.findall(r"\w+(?:['’]\w+)?|\[[^\]]+\]|[.,;:!?]", text, flags = re.UNICODE)


def isWord(token):
    return bool(token) and (token[0].isalnum() or token[0] == "[")


def getWords(text):
    return [token for token in tokenize(text) if isWord(token)]


def normalizeWord(word):
    word = unicodedata.normalize("NFKC", str(word or "")).lower().replace("‘", "'").replace("’", "'")
    if word.startswith("[") and word.endswith("]"):
        word = word[1 : -1]
    return re.sub(r"^[^\w']+|[^\w']+$", "", word, flags = re.UNICODE)


def canonicalWord(word, variantModel = None):
    word = normalizeWord(word)
    if variantModel:
        return variantModel["map"].get(word, word)
    return word


def phraseKey(tokens, variantModel = None):
    return " ".join(word for word in (canonicalWord(token, variantModel) for token in tokens) if word)


def commonPrefix(first, second):
    length = 0
    while length < len(first) and length < len(second) and first[length] == second[length]:
        length += 1
    return length


def commonSuffix(first, second):
    length = 0
    while length < len(first) and length < len(second) and first[-length - 1] == second[-length - 1]:
        length += 1
    return length


def longestCommonSubsequence(first, second):
    scores = [0] * (len(second) + 1)
    for firstLetter in first:
        previous = 0
        for position, secondLetter in enumerate(second, 1):
            oldScore = scores[position]
            if firstLetter == secondLetter:
                scores[position] = previous + 1
            else:
                scores[position] = max(scores[position], scores[position - 1])
            previous = oldScore
    return scores[-1]


def editDistance(first, second):
    previous = list(range(len(second) + 1))
    for firstPosition, firstLetter in enumerate(first, 1):
        current = [firstPosition]
        for secondPosition, secondLetter in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[secondPosition] + 1, previous[secondPosition - 1] + (firstLetter != secondLetter)))
        previous = current
    return previous[-1]


def removePossessive(word):
    if word.endswith("'s") and len(word) > 3:
        return word[: -2]
    if word.endswith("s'") and len(word) > 3:
        return word[: -1]
    return word


def compareWords(first, second):
    first = normalizeWord(first)
    second = normalizeWord(second)
    if not first or not second:
        return {"score": 0.0, "kind": ""}
    if first == second:
        return {"score": 1.0, "kind": "exact"}

    firstBase = removePossessive(first)
    secondBase = removePossessive(second)
    if firstBase == secondBase:
        return {"score": 0.98, "kind": "possessive"}

    prefix = commonPrefix(first, second)
    suffix = commonSuffix(first, second)
    minimumLength = min(len(first), len(second))
    maximumLength = max(len(first), len(second))
    firstTail = len(first) - prefix
    secondTail = len(second) - prefix
    firstHead = len(first) - suffix
    secondHead = len(second) - suffix

    if (first.startswith(second) or second.startswith(first)) and minimumLength >= 4:
        added = maximumLength - minimumLength
        if added <= 2:
            return {"score": 0.93, "kind": "suffix"}
        if added <= 3 and minimumLength >= 5:
            return {"score": 0.84, "kind": "suffix"}

    if (first.endswith(second) or second.endswith(first)) and minimumLength >= 4:
        added = maximumLength - minimumLength
        if added <= 2:
            return {"score": 0.93, "kind": "prefix"}
        if added <= 3 and minimumLength >= 5:
            return {"score": 0.84, "kind": "prefix"}

    if prefix >= 4 and minimumLength >= 5 and firstTail <= 3 and secondTail <= 3 and firstTail + secondTail:
        return {"score": 0.84, "kind": "ending"}

    if suffix >= 4 and minimumLength >= 5 and firstHead <= 3 and secondHead <= 3 and firstHead + secondHead:
        return {"score": 0.84, "kind": "beginning"}

    distance = editDistance(first, second)
    commonLetters = longestCommonSubsequence(first, second)
    if prefix >= 3 and suffix >= 1 and distance <= 2 and commonLetters >= minimumLength - 1:
        return {"score": 0.87, "kind": "spelling"}

    if suffix >= 3 and prefix >= 1 and distance <= 2 and commonLetters >= minimumLength - 1:
        return {"score": 0.82, "kind": "spelling"}

    if minimumLength >= 5 and distance <= max(1, minimumLength // 5) and commonLetters / maximumLength >= 0.8 and prefix + suffix >= 4:
        return {"score": 0.76, "kind": "spelling"}

    return {"score": 0.0, "kind": ""}


def relationKey(first, second):
    return tuple(sorted((normalizeWord(first), normalizeWord(second))))


def buildVariantModel(sentences):
    wordCounts = Counter()
    for sentence in sentences:
        for word in getWords(sentence):
            word = normalizeWord(word)
            if word:
                wordCounts[word] += 1

    words = sorted(wordCounts)
    parent = {word: word for word in words}
    sizes = {word: 1 for word in words}

    def findRoot(word):
        while parent[word] != word:
            parent[word] = parent[parent[word]]
            word = parent[word]
        return word

    def merge(first, second):
        first = findRoot(first)
        second = findRoot(second)
        if first == second:
            return
        if sizes[first] < sizes[second]:
            first, second = second, first
        parent[second] = first
        sizes[first] += sizes[second]

    relations = {}
    for firstPosition, first in enumerate(words):
        for second in words[firstPosition + 1 :]:
            relation = compareWords(first, second)
            if relation["score"]:
                relations[relationKey(first, second)] = relation
                if relation["score"] >= 0.8:
                    merge(first, second)

    rawGroups = defaultdict(list)
    for word in words:
        rawGroups[findRoot(word)].append(word)

    groups = {}
    variantMap = {}
    for members in rawGroups.values():
        groupKey = min(members)
        orderedMembers = sorted(members, key = lambda word: (-wordCounts[word], word))
        groups[groupKey] = orderedMembers
        for member in members:
            variantMap[member] = groupKey

    return {"map": variantMap, "groups": groups, "relations": relations, "counts": wordCounts}


def describeVariants(variantModel):
    descriptions = []
    for groupKey, members in sorted(variantModel["groups"].items()):
        if len(members) < 2:
            continue
        relations = []
        for firstPosition, first in enumerate(members):
            for second in members[firstPosition + 1 :]:
                relation = variantModel["relations"].get(relationKey(first, second))
                if relation:
                    relations.append({"first": first, "second": second, "kind": relation["kind"], "score": relation["score"]})
        descriptions.append({"forms": members, "relations": relations})
    return descriptions


def discoverVariants(sentences):
    sentences = prepareSentences(sentences)
    return describeVariants(buildVariantModel(sentences))


def getRelation(first, second, variantModel = None):
    if variantModel:
        relation = variantModel["relations"].get(relationKey(first, second))
        if relation:
            return relation
        first = normalizeWord(first)
        second = normalizeWord(second)
        if first != second and variantModel["map"].get(first) == variantModel["map"].get(second):
            return {"score": 0.8, "kind": "related"}
    return compareWords(first, second)


def wordScore(first, second, variantModel = None):
    first = normalizeWord(first)
    second = normalizeWord(second)
    if not first or not second:
        return -5.5
    if first == second:
        return 6
    relation = getRelation(first, second, variantModel)
    if relation["score"]:
        return 1.5 + 4 * relation["score"]
    return -5.5


def lcsRatio(first, second):
    scores = [0] * (len(second) + 1)
    for firstWord in first:
        previous = 0
        for position, secondWord in enumerate(second, 1):
            oldScore = scores[position]
            if firstWord == secondWord:
                scores[position] = previous + 1
            else:
                scores[position] = max(scores[position], scores[position - 1])
            previous = oldScore
    return scores[-1] / max(1, len(first), len(second))


def setOverlap(first, second):
    firstSet = set(first)
    secondSet = set(second)
    return len(firstSet & secondSet) / max(1, len(firstSet | secondSet))


def getBigrams(words):
    return set(zip(words, words[1 :]))


def sentenceSimilarity(firstSentence, secondSentence, variantModel = None):
    firstWords = [canonicalWord(word, variantModel) for word in getWords(firstSentence) if canonicalWord(word, variantModel)]
    secondWords = [canonicalWord(word, variantModel) for word in getWords(secondSentence) if canonicalWord(word, variantModel)]
    firstBigrams = getBigrams(firstWords)
    secondBigrams = getBigrams(secondWords)
    bigramOverlap = len(firstBigrams & secondBigrams) / max(1, len(firstBigrams | secondBigrams))
    return 0.4 * setOverlap(firstWords, secondWords) + 0.4 * lcsRatio(firstWords, secondWords) + 0.2 * bigramOverlap


def findMedoid(sentences, variantModel = None):
    totals = []
    for firstIndex, sentence in enumerate(sentences):
        total = sum(sentenceSimilarity(sentence, other, variantModel) for secondIndex, other in enumerate(sentences) if firstIndex != secondIndex)
        totals.append(total)
    medoidIndex = max(range(len(sentences)), key = lambda index: totals[index])
    meanSimilarity = totals[medoidIndex] / (len(sentences) - 1) if len(sentences) > 1 else 1.0
    return medoidIndex, totals, meanSimilarity


def splitClauses(sentence):
    rawClauses = re.split(r"\s*(?:[;:.!?]+|,\s*(?=(?:but|and|yet)\b))\s*", sentence, flags = re.I)
    clauses = []
    for rawClause in rawClauses:
        if not rawClause.strip():
            continue
        for clause in re.split(r"\s+(?=(?:but|yet)\b)", rawClause, flags = re.I):
            clause = re.sub(r"^[,;:.!?\s]+|[,;:.!?\s]+$", "", clause.strip())
            if clause:
                clauses.append(clause)
    return clauses


def alignClauses(referenceClause, sentenceClause, variantModel = None):
    referenceWords = getWords(referenceClause)
    sentenceWords = getWords(sentenceClause)
    referenceLength = len(referenceWords)
    sentenceLength = len(sentenceWords)
    gapPenalty = -2.5
    scores = [[0.0] * (sentenceLength + 1) for position in range(referenceLength + 1)]
    directions = [[""] * (sentenceLength + 1) for position in range(referenceLength + 1)]

    for referencePosition in range(1, referenceLength + 1):
        scores[referencePosition][0] = scores[referencePosition - 1][0] + gapPenalty
        directions[referencePosition][0] = "up"

    for sentencePosition in range(1, sentenceLength + 1):
        scores[0][sentencePosition] = scores[0][sentencePosition - 1] + gapPenalty
        directions[0][sentencePosition] = "left"

    for referencePosition in range(1, referenceLength + 1):
        for sentencePosition in range(1, sentenceLength + 1):
            choices = (
                (scores[referencePosition - 1][sentencePosition - 1] + wordScore(referenceWords[referencePosition - 1], sentenceWords[sentencePosition - 1], variantModel), "diagonal"),
                (scores[referencePosition - 1][sentencePosition] + gapPenalty, "up"),
                (scores[referencePosition][sentencePosition - 1] + gapPenalty, "left")
            )
            scores[referencePosition][sentencePosition], directions[referencePosition][sentencePosition] = max(choices, key = lambda choice: choice[0])

    operations = []
    referencePosition = referenceLength
    sentencePosition = sentenceLength

    while referencePosition or sentencePosition:
        direction = directions[referencePosition][sentencePosition]
        if direction == "diagonal":
            operations.append(("match", referencePosition - 1, sentencePosition - 1))
            referencePosition -= 1
            sentencePosition -= 1
        elif direction == "up":
            operations.append(("delete", referencePosition - 1, -1))
            referencePosition -= 1
        else:
            operations.append(("insert", -1, sentencePosition - 1))
            sentencePosition -= 1

    return referenceWords, sentenceWords, operations[::-1]


def matchClauses(referenceClauses, sentences, variantModel = None):
    groups = [[] for clause in referenceClauses]
    for sentence in sentences:
        sentenceClauses = splitClauses(sentence)
        usedClauses = set()
        for referenceIndex, referenceClause in enumerate(referenceClauses):
            candidates = [(sentenceSimilarity(referenceClause, clause, variantModel), clauseIndex, clause) for clauseIndex, clause in enumerate(sentenceClauses) if clauseIndex not in usedClauses]
            if candidates:
                similarity, clauseIndex, clause = max(candidates, key = lambda candidate: candidate[0])
                usedClauses.add(clauseIndex)
                groups[referenceIndex].append(clause)
            else:
                groups[referenceIndex].append("")
    return groups


def buildProfile(referenceClause, clauses, variantModel = None):
    referenceWords = getWords(referenceClause)
    referenceLength = len(referenceWords)
    rows = []

    for clause in clauses:
        referenceTokens, sentenceTokens, operations = alignClauses(referenceClause, clause, variantModel)
        anchors = [""] * referenceLength
        buckets = [[] for position in range(referenceLength + 1)]
        referencePosition = 0

        for operation, referenceIndex, sentenceIndex in operations:
            if operation == "match":
                anchors[referenceIndex] = sentenceTokens[sentenceIndex]
                referencePosition = referenceIndex + 1
            elif operation == "delete":
                referencePosition = referenceIndex + 1
            else:
                buckets[referencePosition].append(sentenceTokens[sentenceIndex])

        rows.append({"anchors": anchors, "buckets": buckets})

    units = []
    for position in range(referenceLength + 1):
        units.append({"type": "bucket", "position": position, "realisations": [row["buckets"][position] for row in rows]})
        if position < referenceLength:
            units.append({"type": "anchor", "position": position, "realisations": [[row["anchors"][position]] if row["anchors"][position] else [] for row in rows]})
    return units


def chooseSurface(realisations, key, centrality, variantModel = None):
    candidates = [index for index, realisation in enumerate(realisations) if phraseKey(realisation, variantModel) == key]
    if not candidates:
        return []

    exactSurfaces = Counter(" ".join(realisations[index]) for index in candidates)
    highestCount = max(exactSurfaces.values())
    mostCommonSurfaces = {surface for surface, count in exactSurfaces.items() if count == highestCount}

    if len(mostCommonSurfaces) == 1:
        surface = next(iter(mostCommonSurfaces))
        return surface.split(" ") if surface else []

    bestIndex = max((index for index in candidates if " ".join(realisations[index]) in mostCommonSurfaces), key = lambda index: centrality[index])
    return realisations[bestIndex]


def countNgrams(clauses, variantModel = None):
    bigrams = Counter()
    trigrams = Counter()
    for clause in clauses:
        words = [canonicalWord(word, variantModel) for word in getWords(clause) if canonicalWord(word, variantModel)]
        bigrams.update(zip(words, words[1 :]))
        trigrams.update(zip(words, words[1 :], words[2 :]))
    return bigrams, trigrams


def synthesizeClause(referenceClause, clauses, threshold, variantModel = None):
    units = buildProfile(referenceClause, clauses, variantModel)
    clauseCount = len(clauses)
    centrality = [sum(sentenceSimilarity(clause, other, variantModel) for otherIndex, other in enumerate(clauses) if clauseIndex != otherIndex) for clauseIndex, clause in enumerate(clauses)]
    bigrams, trigrams = countNgrams(clauses, variantModel)
    stableUnits = []
    stableTokens = {}

    for unitIndex, unit in enumerate(units):
        if unit["type"] != "anchor":
            continue
        counts = Counter(phraseKey(realisation, variantModel) for realisation in unit["realisations"])
        key, count = max(counts.items(), key = lambda item: item[1])
        if key and count / clauseCount >= threshold:
            stableUnits.append(unitIndex)
            stableTokens[unitIndex] = chooseSurface(unit["realisations"], key, centrality, variantModel)

    def getRange(sentenceIndex, start, end):
        tokens = []
        for unitIndex in range(start, end):
            tokens.extend(units[unitIndex]["realisations"][sentenceIndex])
        return tokens

    boundaries = [-1, *stableUnits, len(units)]
    output = []

    for boundaryIndex in range(len(boundaries) - 1):
        leftBoundary = boundaries[boundaryIndex]
        rightBoundary = boundaries[boundaryIndex + 1]
        realisations = [getRange(sentenceIndex, leftBoundary + 1, rightBoundary) for sentenceIndex in range(clauseCount)]
        groups = defaultdict(list)

        for sentenceIndex, realisation in enumerate(realisations):
            groups[phraseKey(realisation, variantModel)].append(sentenceIndex)

        leftTokens = stableTokens.get(leftBoundary, [])
        rightTokens = stableTokens.get(rightBoundary, [])
        bestChoice = None

        for key, sourceIndexes in groups.items():
            representative = max(sourceIndexes, key = lambda sentenceIndex: centrality[sentenceIndex])
            tokens = realisations[representative]
            canonicalTokens = [canonicalWord(token, variantModel) for token in tokens if canonicalWord(token, variantModel)]
            score = len(sourceIndexes) * 5 + centrality[representative]
            score += sum(1.3 * math.log1p(bigrams[(first, second)]) for first, second in zip(canonicalTokens, canonicalTokens[1 :]))
            score += sum(1.8 * math.log1p(trigrams[(first, second, third)]) for first, second, third in zip(canonicalTokens, canonicalTokens[1 :], canonicalTokens[2 :]))

            if leftTokens and canonicalTokens:
                count = bigrams[(canonicalWord(leftTokens[-1], variantModel), canonicalTokens[0])]
                score += 3 * math.log1p(count) if count else -4

            if canonicalTokens and rightTokens:
                count = bigrams[(canonicalTokens[-1], canonicalWord(rightTokens[0], variantModel))]
                score += 3 * math.log1p(count) if count else -4

            if not canonicalTokens and leftTokens and rightTokens:
                count = bigrams[(canonicalWord(leftTokens[-1], variantModel), canonicalWord(rightTokens[0], variantModel))]
                score += 3 * math.log1p(count) if count else -3

            if (not leftTokens or not rightTokens) and canonicalTokens:
                score -= 1.5 * len(canonicalTokens)

            if bestChoice is None or score > bestChoice[0]:
                bestChoice = (score, tokens)

        if bestChoice:
            output.extend(bestChoice[1])
        if rightBoundary in stableTokens:
            output.extend(stableTokens[rightBoundary])

    surfaceCounts = defaultdict(Counter)
    for clause in clauses:
        for token in getWords(clause):
            surfaceCounts[canonicalWord(token, variantModel)][token] += 1

    result = []
    for token in output:
        canonical = canonicalWord(token, variantModel)
        surfaces = surfaceCounts.get(canonical)
        if not surfaces:
            result.append(token)
            continue

        surface = max(surfaces.items(), key = lambda item: item[1])[0]
        if token.startswith("[") and token.endswith("]"):
            plainSurfaces = [(text, count) for text, count in surfaces.items() if not (text.startswith("[") and text.endswith("]"))]
            if plainSurfaces:
                result.append(max(plainSurfaces, key = lambda item: item[1])[0])
                continue

        plainToken = normalizeWord(token)
        result.append(surface if canonical != plainToken else token)

    return " ".join(result)


def prepareSentences(sentences):
    if isinstance(sentences, str):
        sentences = re.split(r"\n+", sentences)
    return [str(sentence).strip() for sentence in sentences if str(sentence).strip()]


def synthesizeSentences(sentences, mode = "balanced", returnDiagnostics = False):
    sentences = prepareSentences(sentences)

    if not sentences:
        diagnostics = {"text": "", "medoid": None, "meanSimilarity": 0.0, "clauses": 0, "reference": "", "variants": []}
        return diagnostics if returnDiagnostics else ""

    if mode not in modeThresholds:
        raise ValueError("mode must be 'conservative', 'balanced', or 'exploratory'")

    variantModel = buildVariantModel(sentences)
    if len(sentences) == 1:
        diagnostics = {"text": sentences[0], "medoid": 0, "meanSimilarity": 1.0, "clauses": 1, "reference": sentences[0], "variants": describeVariants(variantModel)}
        return diagnostics if returnDiagnostics else diagnostics["text"]

    medoidIndex, totals, meanSimilarity = findMedoid(sentences, variantModel)
    reference = sentences[medoidIndex]
    referenceClauses = splitClauses(reference)
    clauseGroups = matchClauses(referenceClauses, sentences, variantModel)
    parts = [synthesizeClause(referenceClause, clauseGroups[index], modeThresholds[mode], variantModel).strip() for index, referenceClause in enumerate(referenceClauses)]
    parts = [part for part in parts if part]

    result = ""
    for index, part in enumerate(parts):
        if index == 0:
            result = part
        elif re.match(r"^(but|and|yet)\b", part, re.I):
            result += ", " + part
        else:
            result += "; " + part

    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"\[\s+", "[", result)
    result = re.sub(r"\s+\]", "]", result)

    if result:
        result = result[0].upper() + result[1 :]

    punctuationMatch = re.search(r"[.!?]\s*$", reference.strip())
    punctuation = punctuationMatch.group(0).strip() if punctuationMatch else "."
    result = re.sub(r"[.!?]+$", "", result) + punctuation

    diagnostics = {"text": result, "medoid": medoidIndex, "meanSimilarity": meanSimilarity, "clauses": len(referenceClauses), "reference": reference, "variants": describeVariants(variantModel)}
    return diagnostics if returnDiagnostics else result

if __name__=='__main__':

    example="""The king’s favour [is] toward a wise servant: but his wrath is [against] him that causeth shame.
The king’s favor is toward a servant that dealeth wisely; But his wrath will be against him that causeth shame.
A king delights in a wise servant, but his anger falls on the shameful.
An intelligent minister is acceptable to the king. Whoever is useless shall bear his wrath.
A wise servant is acceptable to the king: he that is good for nothing shall feel his anger.
The king’s favour is toward a servant that dealeth wisely; But his wrath striketh him that dealeth shamefully.
The acceptance of the king to a servant of understanding: and his wrath shall be to him causing shame.
The king’s favor is toward a servant who deals wisely, but his wrath is toward one who causes shame.
The favour of a king is to a wise servant, And an object of his wrath is one causing shame!
The king favors the wise servant, but his wrath falls upon the one who brings disgrace.
The king favors the servant who acts wisely; his wrath, however, is directed against the one who brings disgrace.
A king takes pleasure in a wise servant, but his wrath strikes the disgraceful one.
A discerning servant is pleasing to the king; but he who is good for nothing feels his wrath.
A wise servant is pleasing to the king; he who is of no use feels his wrath.
The king favors the servant who acts wisely; his wrath, however, falls upon the one who acts disgracefully.
The king grants his favor to a discerning servant, but the one who brings disgrace feels his wrath.
The king favors the servant who acts wisely, but his wrath is directed against the one who brings disgrace.
The king favors a wise servant, and the target of his wrath is the one who brings disgrace!!"""
    print(synthesizeSentences(example,'balanced'))
