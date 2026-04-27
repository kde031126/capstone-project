import { Audio } from 'expo-av';
import { useLocalSearchParams, useRouter } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import { Alert, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';

type WordExercise = {
  word: string;
  emoji: string;
  sound: string;
  hint: string;
  focus: string;
  praise: string;
  feedback: string;
};

type Lesson = {
  id: string;
  title: string;
  subtitle: string;
  emoji: string;
  words: WordExercise[];
};

const lessons: Lesson[] = [
  {
    id: 'lesson-1',
    title: '1단계 · 과일 길',
    subtitle: '기초 낱말을 듣고 또박또박 말해요.',
    emoji: '🍎',
    words: [
      {
        word: '사과',
        emoji: '🍎',
        sound: '사-과',
        hint: '입을 크게 열고 천천히 따라 말해요.',
        focus: '과',
        praise: '정말 잘했어요!',
        feedback: '“과” 부분을 조금 더 또렷하게 말하면 더 멋진 발음이 돼요.',
      },
      {
        word: '포도',
        emoji: '🍇',
        sound: '포-도',
        hint: '두 음절을 또박또박 이어서 말해요.',
        focus: '포',
        praise: '좋은 시작이에요!',
        feedback: '처음 “포” 소리를 조금 더 크게 내면 훨씬 선명해져요.',
      },
      {
        word: '딸기',
        emoji: '🍓',
        sound: '딸-기',
        hint: '딸 소리를 짧고 힘 있게 말해요.',
        focus: '딸',
        praise: '아주 좋아요!',
        feedback: '“딸” 부분을 조금 더 힘 있게 말하면 더 정확해져요.',
      },
      {
        word: '바나나',
        emoji: '🍌',
        sound: '바-나-나',
        hint: '리듬 있게 세 번 나누어 말해요.',
        focus: '나',
        praise: '끝까지 잘 따라왔어요!',
        feedback: '가운데 “나”를 한 번 더 또박또박 말하면 훨씬 좋아져요.',
      },
      {
        word: '수박',
        emoji: '🍉',
        sound: '수-박',
        hint: '마지막 박 소리를 또렷하게 마무리해요.',
        focus: '박',
        praise: '멋지게 마무리했어요!',
        feedback: '“박” 받침을 살짝 더 분명히 말하면 더 좋아요.',
      },
    ],
  },
  {
    id: 'lesson-2',
    title: '2단계 · 동물 숲',
    subtitle: '귀여운 동물 이름을 말해봐요.',
    emoji: '🐰',
    words: [
      {
        word: '토끼',
        emoji: '🐰',
        sound: '토-끼',
        hint: '마지막 끼 소리를 귀엽게 또렷하게 말해요.',
        focus: '끼',
        praise: '좋은 도전이었어요!',
        feedback: '“끼” 부분을 조금 더 힘 있게 말하면 더 정확해져요.',
      },
      {
        word: '고양이',
        emoji: '🐱',
        sound: '고-양-이',
        hint: '세 음절을 부드럽게 이어 말해요.',
        focus: '양',
        praise: '정말 잘 따라 했어요!',
        feedback: '가운데 “양” 소리를 조금 더 길게 말하면 더 자연스러워요.',
      },
      {
        word: '강아지',
        emoji: '🐶',
        sound: '강-아-지',
        hint: '처음 강 소리를 또렷하게 말해요.',
        focus: '강',
        praise: '계속 좋아지고 있어요!',
        feedback: '“강” 부분을 조금 더 또렷하게 말하면 훨씬 좋아져요.',
      },
      {
        word: '곰',
        emoji: '🐻',
        sound: '곰',
        hint: '짧지만 굵게 말해봐요.',
        focus: '곰',
        praise: '짧은 단어도 잘했어요!',
        feedback: '입모양을 둥글게 하며 “곰”을 말하면 더 정확해져요.',
      },
      {
        word: '새',
        emoji: '🐦',
        sound: '새',
        hint: '밝고 가볍게 말해요.',
        focus: '새',
        praise: '상큼하게 잘 말했어요!',
        feedback: '“새” 소리를 조금 더 맑게 내면 더 예쁜 발음이 돼요.',
      },
    ],
  },
  {
    id: 'lesson-3',
    title: '3단계 · 음식 마을',
    subtitle: '맛있는 음식 단어를 연습해요.',
    emoji: '🍚',
    words: [
      {
        word: '우유',
        emoji: '🥛',
        sound: '우-유',
        hint: '두 음절을 또박또박 말해요.',
        focus: '유',
        praise: '아주 좋아요!',
        feedback: '마지막 “유” 소리를 조금 더 길게 말하면 더 자연스러워요.',
      },
      {
        word: '빵',
        emoji: '🍞',
        sound: '빵',
        hint: '짧고 통통 튀게 말해요.',
        focus: '빵',
        praise: '통통 튀는 발음이었어요!',
        feedback: '“빵”의 받침을 조금 더 또렷하게 말하면 더 좋아요.',
      },
      {
        word: '주스',
        emoji: '🧃',
        sound: '주-스',
        hint: '주와 스를 또렷하게 나눠 말해요.',
        focus: '스',
        praise: '잘했어요!',
        feedback: '마지막 “스” 소리를 살짝 더 분명하게 말해보세요.',
      },
      {
        word: '국수',
        emoji: '🍜',
        sound: '국-수',
        hint: '국 소리를 짧게, 수는 부드럽게 말해요.',
        focus: '국',
        praise: '참 잘하고 있어요!',
        feedback: '“국” 부분의 ㄱ 소리를 조금 더 살리면 더 정확해져요.',
      },
      {
        word: '밥',
        emoji: '🍚',
        sound: '밥',
        hint: '짧고 단단하게 말해요.',
        focus: '밥',
        praise: '정말 멋져요!',
        feedback: '받침 “ㅂ”을 조금 더 닫아주면 더 또렷해져요.',
      },
    ],
  },
  {
    id: 'lesson-4',
    title: '4단계 · 집 안 탐험',
    subtitle: '생활 속 단어를 익혀봐요.',
    emoji: '🏠',
    words: [
      {
        word: '문',
        emoji: '🚪',
        sound: '문',
        hint: '짧게 한 번 또렷하게 말해요.',
        focus: '문',
        praise: '또박또박 잘했어요!',
        feedback: '“문”의 ㅁ 소리를 조금 더 선명하게 말하면 좋아요.',
      },
      {
        word: '의자',
        emoji: '🪑',
        sound: '의-자',
        hint: '첫 음절을 천천히 말해요.',
        focus: '의',
        praise: '차분하게 잘했어요!',
        feedback: '처음 “의” 부분을 조금 더 부드럽게 말하면 좋아요.',
      },
      {
        word: '책상',
        emoji: '📚',
        sound: '책-상',
        hint: '책 소리를 또렷하게 꺼내요.',
        focus: '책',
        praise: '점점 더 좋아져요!',
        feedback: '“책” 부분을 조금 더 분명하게 말하면 더 멋져요.',
      },
      {
        word: '창문',
        emoji: '🪟',
        sound: '창-문',
        hint: '창과 문을 끊어 읽어도 좋아요.',
        focus: '창',
        praise: '좋았어요!',
        feedback: '처음 “창” 소리를 조금 더 크게 내면 더 좋아요.',
      },
      {
        word: '시계',
        emoji: '⏰',
        sound: '시-계',
        hint: '시 소리를 가볍게 말해요.',
        focus: '계',
        praise: '잘 마무리했어요!',
        feedback: '마지막 “계” 부분을 조금 더 또렷하게 말하면 좋아요.',
      },
    ],
  },
  {
    id: 'lesson-5',
    title: '5단계 · 학교 놀이터',
    subtitle: '학교에서 쓰는 말을 연습해요.',
    emoji: '🎒',
    words: [
      {
        word: '학교',
        emoji: '🏫',
        sound: '학-교',
        hint: '학과 교를 분명히 나누어 말해요.',
        focus: '교',
        praise: '대단해요!',
        feedback: '“교” 부분을 조금 더 길게 말하면 더 자연스러워요.',
      },
      {
        word: '공책',
        emoji: '📒',
        sound: '공-책',
        hint: '받침이 들리게 말해요.',
        focus: '책',
        praise: '아주 멋졌어요!',
        feedback: '마지막 “책”을 조금 더 또렷하게 마무리해보세요.',
      },
      {
        word: '연필',
        emoji: '✏️',
        sound: '연-필',
        hint: '연 소리를 부드럽게 시작해요.',
        focus: '필',
        praise: '훌륭해요!',
        feedback: '“필” 부분의 ㅍ 소리를 살짝 더 강조하면 좋아요.',
      },
      {
        word: '친구',
        emoji: '🧑‍🤝‍🧑',
        sound: '친-구',
        hint: '친과 구를 웃으며 말해요.',
        focus: '친',
        praise: '기분 좋게 잘했어요!',
        feedback: '처음 “친” 부분을 조금 더 밝게 말하면 더 예뻐요.',
      },
      {
        word: '선생님',
        emoji: '👩‍🏫',
        sound: '선-생-님',
        hint: '세 음절을 천천히 이어 말해요.',
        focus: '생',
        praise: '정말 훌륭해요!',
        feedback: '가운데 “생” 부분을 조금 더 또렷하게 말하면 완벽해져요.',
      },
    ],
  },
];

function getText(value: string | string[] | undefined, fallback: string) {
  if (Array.isArray(value)) {
    return value[0] ?? fallback;
  }

  return value ?? fallback;
}

export default function RoadmapExerciseScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    childName?: string | string[];
  }>();
  const [selectedLessonIndex, setSelectedLessonIndex] = useState<number | null>(null);
  const [lessonProgress, setLessonProgress] = useState<number[]>(() => lessons.map(() => 0));
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [recordingKey, setRecordingKey] = useState<string | null>(null);
  const [speakingKey, setSpeakingKey] = useState<string | null>(null);
  const [feedbackByKey, setFeedbackByKey] = useState<Record<string, string>>({});
  const [recordedUris, setRecordedUris] = useState<Record<string, string>>({});
  const playbackRef = useRef<Audio.Sound | null>(null);

  const childName = getText(params.childName, '친구');
  const selectedLesson = selectedLessonIndex !== null ? lessons[selectedLessonIndex] : null;
  const currentProgress = selectedLessonIndex !== null ? lessonProgress[selectedLessonIndex] : 0;
  const lessonCompleted = Boolean(selectedLesson && currentProgress >= selectedLesson.words.length);
  const currentWord = selectedLesson && !lessonCompleted ? selectedLesson.words[currentProgress] : null;
  const currentWordKey = selectedLesson && currentWord ? `${selectedLesson.id}-${currentProgress}` : null;
  const currentFeedback = currentWordKey ? feedbackByKey[currentWordKey] : '';
  const hasCurrentRecording = currentWordKey ? Boolean(recordedUris[currentWordKey]) : false;

  useEffect(() => {
    return () => {
      playbackRef.current?.unloadAsync().catch(() => undefined);
    };
  }, []);

  const isLessonUnlocked = (index: number) => {
    if (index === 0) {
      return true;
    }

    return lessonProgress[index - 1] >= lessons[index - 1].words.length;
  };

  const openLesson = (index: number) => {
    if (!isLessonUnlocked(index)) {
      Alert.alert('아직 잠겨 있어요', '이전 원의 5개 단어를 모두 끝내면 열려요.');
      return;
    }

    setSelectedLessonIndex(index);
  };

  const handleGoBack = () => {
    if (selectedLessonIndex !== null) {
      setSelectedLessonIndex(null);
      return;
    }

    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace('/welcome');
  };

  const handleListen = async (word: WordExercise, key: string) => {
    setSpeakingKey(key);

    try {
      await Speech.stop();
      Speech.speak(word.word, {
        language: 'ko-KR',
        rate: 0.78,
        pitch: 1.05,
        onDone: () => setSpeakingKey(null),
        onStopped: () => setSpeakingKey(null),
        onError: () => setSpeakingKey(null),
      });
    } catch {
      setSpeakingKey(null);
      Alert.alert('듣기 기능을 열 수 없어요', '잠시 후 다시 시도해주세요.');
    }
  };

  const startRecording = async (word: WordExercise, key: string) => {
    if (recording) {
      Alert.alert('녹음 중이에요', '먼저 현재 녹음을 완료해주세요.');
      return;
    }

    try {
      const permission = await Audio.requestPermissionsAsync();

      if (permission.status !== 'granted') {
        Alert.alert('마이크 권한이 필요해요', '아이 목소리를 녹음하려면 마이크를 허용해주세요.');
        return;
      }

      await Speech.stop();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const result = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      setRecording(result.recording);
      setRecordingKey(key);
    } catch {
      Alert.alert('녹음을 시작할 수 없어요', '다시 한 번 눌러서 시도해주세요.');
    }
  };

const stopRecording = async (word: WordExercise, key: string) => {
  if (!recording) return;

  try {
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    if (!uri) return;

    setRecordedUris((prev) => ({ ...prev, [key]: uri }));

    // ✅ 수정된 FormData 구성 방식
    const formData = new FormData();
    formData.append("session_id", "1");
    formData.append("word_id", "1");

    // ReactNative에서는 객체 형태로 파일 정보를 넣어주는 것이 표준입니다.
    formData.append("audio_file", {
      uri: uri,
      type: "audio/x-wav", // 또는 'audio/m4a'
      name: "recording.wav", // 백엔드에서 인식하기 가장 쉬운 wav로 명시
    } as any);

    const res = await fetch("http://10.240.82.63:8000/records/", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      console.log("UPLOAD ERROR:", res.status, data);
    } else {
      console.log("UPLOAD SUCCESS:", data);
    }

    setFeedbackByKey((prev) => ({
      ...prev,
      [key]: word.feedback,
    }));
  } catch (err) {
    console.log("RECORD ERROR:", err);
    Alert.alert("녹음을 마치지 못했어요", "다시 시도해주세요.");
  } finally {
    setRecording(null);
    setRecordingKey(null);
  }
};

  const playMyVoice = async (key: string) => {
    const uri = recordedUris[key];

    if (!uri) {
      return;
    }

    try {
      if (playbackRef.current) {
        await playbackRef.current.unloadAsync();
      }

      const result = await Audio.Sound.createAsync({ uri });
      playbackRef.current = result.sound;
      result.sound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          result.sound.unloadAsync().catch(() => undefined);
          playbackRef.current = null;
        }
      });
      await result.sound.playAsync();
    } catch {
      Alert.alert('재생할 수 없어요', '녹음한 목소리를 다시 확인해주세요.');
    }
  };

  const handleNextWord = () => {
    if (selectedLessonIndex === null || !selectedLesson) {
      return;
    }

    setLessonProgress((prev) => {
      const next = [...prev];
      next[selectedLessonIndex] = Math.min(next[selectedLessonIndex] + 1, selectedLesson.words.length);
      return next;
    });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <Pressable style={styles.backButton} onPress={handleGoBack}>
          <Text style={styles.backButtonText}>
            {selectedLesson ? '← 로드맵으로' : '← 뒤로가기'}
          </Text>
        </Pressable>

        {selectedLesson ? (
          <>
            <View style={styles.lessonHeader}>
              <Text style={styles.lessonBadge}>{selectedLesson.title}</Text>
              <Text style={styles.lessonTitle}>{selectedLesson.subtitle}</Text>
              <Text style={styles.lessonProgressText}>
                {lessonCompleted ? '5 / 5 단어 완료!' : `${currentProgress + 1} / ${selectedLesson.words.length} 단어`}
              </Text>
            </View>

            {lessonCompleted ? (
              <View style={styles.completeCard}>
                <Text style={styles.completeEmoji}>⭐</Text>
                <Text style={styles.completeTitle}>이 단계를 모두 끝냈어요!</Text>
                <Text style={styles.completeText}>
                  정말 잘했어요. 다음 원이 열렸으니 새로운 단어도 도전해보세요.
                </Text>
                <Pressable style={styles.nextButton} onPress={() => setSelectedLessonIndex(null)}>
                  <Text style={styles.nextButtonText}>로드맵으로 돌아가기</Text>
                </Pressable>
              </View>
            ) : currentWord && currentWordKey ? (
              <View style={styles.exerciseCard}>
                <View style={styles.wordRow}>
                  <View style={styles.wordCircle}>
                    <Text style={styles.wordEmoji}>{currentWord.emoji}</Text>
                  </View>
                  <View style={styles.wordTextWrap}>
                    <Text style={styles.word}>{currentWord.word}</Text>
                    <Text style={styles.sound}>{currentWord.sound}</Text>
                  </View>
                </View>

                <Text style={styles.hint}>{currentWord.hint}</Text>

                <View style={styles.actionRow}>
                  <Pressable style={styles.listenButton} onPress={() => handleListen(currentWord, currentWordKey)}>
                    <Text style={styles.listenButtonText}>
                      {speakingKey === currentWordKey ? '듣는 중...' : '🔊 듣기'}
                    </Text>
                  </Pressable>
                  <Pressable
                    style={[styles.recordButton, recordingKey === currentWordKey && styles.recordingButton]}
                    onPress={() =>
                      recordingKey === currentWordKey
                        ? stopRecording(currentWord, currentWordKey)
                        : startRecording(currentWord, currentWordKey)
                    }>
                    <Text style={styles.recordButtonText}>
                      {recordingKey === currentWordKey ? '⏹ 녹음 완료' : '🎙️ 녹음하기'}
                    </Text>
                  </Pressable>
                </View>

                {currentFeedback ? (
                  <View style={styles.feedbackCard}>
                    <Text style={styles.feedbackTitle}>{currentWord.praise}</Text>
                    <Text style={styles.feedbackText}>{currentFeedback}</Text>
                    <Text style={styles.feedbackText}>
                      이번에는 <Text style={styles.focusText}>{currentWord.focus}</Text> 부분에 조금 더 신경 쓰면 더
                      좋아요.
                    </Text>

                    {hasCurrentRecording ? (
                      <Pressable style={styles.playbackButton} onPress={() => playMyVoice(currentWordKey)}>
                        <Text style={styles.playbackButtonText}>▶ 내 목소리 듣기</Text>
                      </Pressable>
                    ) : null}

                    <Pressable style={styles.nextButton} onPress={handleNextWord}>
                      <Text style={styles.nextButtonText}>
                        {currentProgress === selectedLesson.words.length - 1 ? '단계 완료하기' : '다음 단어'}
                      </Text>
                    </Pressable>
                  </View>
                ) : null}
              </View>
            ) : null}
          </>
        ) : (
          <>
            <View style={styles.mapHeader}>
              <Text style={styles.eyebrow}>{childName}의 발음 여행</Text>
              <Text style={styles.title}>로드맵을 따라 하나씩 연습해요</Text>
              <Text style={styles.subtitle}>
                원 하나마다 단어 5개가 들어 있어요. 첫 번째 원부터 순서대로 시작해보세요.
              </Text>
            </View>

            <View style={styles.roadmapWrap}>
              {lessons.map((lesson, index) => {
                const unlocked = isLessonUnlocked(index);
                const completed = lessonProgress[index] >= lesson.words.length;

                return (
                  <View
                    key={lesson.id}
                    style={[
                      styles.nodeRow,
                      index % 2 === 0 ? styles.nodeLeft : styles.nodeRight,
                    ]}>
                    {index < lessons.length - 1 ? <View style={styles.connector} /> : null}

                    <Pressable
                      style={[
                        styles.lessonCircle,
                        !unlocked && styles.lockedCircle,
                        completed && styles.completedCircle,
                      ]}
                      onPress={() => openLesson(index)}>
                      <Text style={styles.lessonEmoji}>{unlocked ? lesson.emoji : '🔒'}</Text>
                      <Text style={styles.lessonNumber}>{index + 1}</Text>
                    </Pressable>

                    <View style={styles.lessonInfoCard}>
                      <Text style={styles.lessonInfoTitle}>{lesson.title}</Text>
                      <Text style={styles.lessonInfoText}>
                        {completed
                          ? '완료! 다음 원이 열렸어요.'
                          : unlocked
                            ? '단어 5개 연습하기'
                            : '이전 원을 끝내면 열려요'}
                      </Text>
                    </View>
                  </View>
                );
              })}
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F9FBF4',
  },
  container: {
    padding: 20,
    gap: 14,
    paddingBottom: 28,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingVertical: 4,
  },
  backButtonText: {
    color: '#59B800',
    fontSize: 14,
    fontWeight: '800',
  },
  mapHeader: {
    backgroundColor: '#63D400',
    borderRadius: 22,
    padding: 18,
    gap: 6,
  },
  eyebrow: {
    color: '#ECFFE0',
    fontSize: 14,
    fontWeight: '800',
  },
  title: {
    fontSize: 28,
    lineHeight: 34,
    fontWeight: '900',
    color: '#FFFFFF',
  },
  subtitle: {
    fontSize: 14,
    lineHeight: 21,
    color: '#F3FFE8',
  },
  roadmapWrap: {
    paddingTop: 8,
    paddingBottom: 8,
    gap: 8,
  },
  nodeRow: {
    width: '92%',
    minHeight: 118,
    position: 'relative',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  nodeLeft: {
    alignSelf: 'flex-start',
  },
  nodeRight: {
    alignSelf: 'flex-end',
    flexDirection: 'row-reverse',
  },
  connector: {
    position: 'absolute',
    left: 42,
    top: 84,
    width: 4,
    height: 62,
    backgroundColor: '#D9E4C4',
    borderRadius: 999,
  },
  lessonCircle: {
    width: 86,
    height: 86,
    borderRadius: 43,
    backgroundColor: '#7BDC10',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 5,
    borderColor: '#B7F26C',
  },
  lockedCircle: {
    backgroundColor: '#E5E7EB',
    borderColor: '#D1D5DB',
  },
  completedCircle: {
    backgroundColor: '#FACC15',
    borderColor: '#FDE68A',
  },
  lessonEmoji: {
    fontSize: 30,
  },
  lessonNumber: {
    marginTop: 2,
    fontSize: 13,
    fontWeight: '900',
    color: '#1F2937',
  },
  lessonInfoCard: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 12,
    gap: 4,
  },
  lessonInfoTitle: {
    fontSize: 16,
    fontWeight: '900',
    color: '#1F2544',
  },
  lessonInfoText: {
    fontSize: 13,
    lineHeight: 18,
    color: '#5C677D',
  },
  lessonHeader: {
    backgroundColor: '#63D400',
    borderRadius: 22,
    padding: 18,
    gap: 6,
  },
  lessonBadge: {
    color: '#E9FFDA',
    fontSize: 14,
    fontWeight: '800',
  },
  lessonTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    lineHeight: 30,
    fontWeight: '900',
  },
  lessonProgressText: {
    color: '#F5FFE7',
    fontSize: 14,
    fontWeight: '700',
  },
  exerciseCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    padding: 16,
    gap: 12,
  },
  wordRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  wordCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#F1FBE7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  wordEmoji: {
    fontSize: 36,
  },
  wordTextWrap: {
    flex: 1,
  },
  word: {
    fontSize: 30,
    fontWeight: '900',
    color: '#22223B',
  },
  sound: {
    fontSize: 16,
    fontWeight: '700',
    color: '#59B800',
    marginTop: 2,
  },
  hint: {
    fontSize: 14,
    lineHeight: 20,
    color: '#5C677D',
  },
  actionRow: {
    flexDirection: 'row',
    gap: 10,
  },
  listenButton: {
    flex: 1,
    backgroundColor: '#EEF9E3',
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: 'center',
  },
  listenButtonText: {
    color: '#468D00',
    fontSize: 14,
    fontWeight: '800',
  },
  recordButton: {
    flex: 1,
    backgroundColor: '#63D400',
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: 'center',
  },
  recordingButton: {
    backgroundColor: '#F25F5C',
  },
  recordButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '800',
  },
  feedbackCard: {
    backgroundColor: '#F7FCEB',
    borderRadius: 16,
    padding: 14,
    gap: 8,
  },
  feedbackTitle: {
    fontSize: 15,
    fontWeight: '900',
    color: '#468D00',
  },
  feedbackText: {
    fontSize: 14,
    lineHeight: 20,
    color: '#4B5563',
  },
  focusText: {
    fontWeight: '900',
    color: '#468D00',
  },
  playbackButton: {
    alignSelf: 'flex-start',
    marginTop: 2,
    backgroundColor: '#FFFFFF',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  playbackButtonText: {
    color: '#468D00',
    fontSize: 13,
    fontWeight: '800',
  },
  nextButton: {
    backgroundColor: '#63D400',
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 4,
  },
  nextButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  completeCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    padding: 18,
    alignItems: 'center',
    gap: 8,
  },
  completeEmoji: {
    fontSize: 42,
  },
  completeTitle: {
    fontSize: 22,
    fontWeight: '900',
    color: '#1F2544',
  },
  completeText: {
    fontSize: 14,
    lineHeight: 21,
    color: '#5C677D',
    textAlign: 'center',
  },
});
