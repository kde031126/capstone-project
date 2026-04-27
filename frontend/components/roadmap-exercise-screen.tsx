import { Audio } from 'expo-av';
import { useLocalSearchParams, useRouter } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useState } from 'react';
import { Alert, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';

// API 함수 불러오기
import { startSession, uploadRecord, endSession } from '../api/sessions';

const lessonInfo = [
  { title: '1단계 · 과일 길', subtitle: '기초 낱말 연습', emoji: '🍎' },
  { title: '2단계 · 동물 숲', subtitle: '동물 이름 말하기', emoji: '🐰' },
  { title: '3단계 · 음식 마을', subtitle: '맛있는 음식 단어', emoji: '🍚' },
  { title: '4단계 · 집 안 탐험', subtitle: '우리 집 물건들', emoji: '🏠' },
  { title: '5단계 · 학교 놀이터', subtitle: '학교 생활 단어', emoji: '🎒' },
];

export default function RoadmapExerciseScreen() {
  const params = useLocalSearchParams<{ childName?: string }>();
  const childName = params.childName || '친구';

  // 상태 관리
  const [selectedLessonIndex, setSelectedLessonIndex] = useState<number | null>(null);
  const [session, setSession] = useState<any>(null); // 서버 응답 저장
  const [currentProgress, setCurrentProgress] = useState(0);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [feedback, setFeedback] = useState<any>(null);

  // 1. 학습 시작 (DB 연동)
  const openLesson = async (index: number) => {
    try {
      const data = await startSession({
        user_id: 1, 
        step_id: index + 1,
        words_count: 5
      });
      
      // 서버에서 target_words가 제대로 오는지 확인
      if (data && data.target_words) {
        setSession(data);
        setSelectedLessonIndex(index);
        setCurrentProgress(0);
        setFeedback(null);
      } else {
        Alert.alert('알림', '학습 단어 정보를 가져오지 못했습니다.');
      }
    } catch (error) {
      Alert.alert('연결 실패', '주소가 /sessions/sessions/ 가 맞는지 확인해주세요.');
    }
  };

  // 2. 녹음 로직 (꾹 누르기)
  const startRecording = async () => {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') return;
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording: newRecording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      setRecording(newRecording);
    } catch (err) { console.error(err); }
  };

  const stopRecording = async () => {
    if (!recording || !session) return;
    setIsAnalyzing(true);
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      if (!uri) return;

      // target_words에서 현재 단어 ID 추출
      const currentWordId = session.target_words[currentProgress].word_id;
      const result = await uploadRecord(session.session_id, currentWordId, uri);
      setFeedback(result);
    } catch (err) {
      Alert.alert('분석 오류', 'AI 분석 서버 응답이 없습니다.');
    } finally {
      setRecording(null);
      setIsAnalyzing(false);
    }
  };

  // 3. 다음 단계 이동
  const handleNextWord = async () => {
    if (currentProgress < session.target_words.length - 1) {
      setCurrentProgress(prev => prev + 1);
      setFeedback(null);
    } else {
      await endSession(session.session_id);
      Alert.alert('참 잘했어요!', '오늘의 연습을 모두 끝냈습니다.', [
        { text: '확인', onPress: () => setSelectedLessonIndex(null) }
      ]);
    }
  };

  // 현재 단어 데이터 안전하게 가져오기
  const currentWord = session?.target_words?.[currentProgress];

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <Pressable style={styles.backButton} onPress={() => setSelectedLessonIndex(null)}>
          <Text style={styles.backButtonText}>← {selectedLessonIndex !== null ? '로드맵' : '뒤로'}</Text>
        </Pressable>

        {selectedLessonIndex !== null && currentWord ? (
          <View style={styles.exerciseCard}>
            <View style={styles.header}>
              <Text style={styles.badge}>{lessonInfo[selectedLessonIndex].title}</Text>
              <Text style={styles.progressText}>{currentProgress + 1} / {session.target_words.length}</Text>
            </View>

            <View style={styles.wordArea}>
              <View style={styles.wordCircle}>
                <Text style={styles.wordEmoji}>{lessonInfo[selectedLessonIndex].emoji}</Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.wordText}>{currentWord.word_text}</Text>
              </View>
            </View>

            <View style={styles.btnRow}>
              <Pressable style={styles.subBtn} onPress={() => Speech.speak(currentWord.word_text, { language: 'ko-KR' })}>
                <Text style={styles.subBtnText}>🔊 듣기</Text>
              </Pressable>
              <Pressable 
                style={[styles.mainBtn, recording && styles.recordingBtn]}
                onPressIn={startRecording}
                onPressOut={stopRecording}
              >
                <Text style={styles.mainBtnText}>
                  {isAnalyzing ? 'AI 분석 중...' : recording ? '말하는 중...' : '🎙️ 꾹 누르기'}
                </Text>
              </Pressable>
            </View>

            {feedback && (
              <View style={styles.feedbackContainer}>
                <Text style={styles.feedbackTitle}>{feedback.is_correct ? '✅ 정답이에요!' : '❌ 다시 해볼까?'}</Text>
                <Text style={styles.feedbackSub}>인식 결과: {feedback.recognized_text}</Text>
                <Pressable style={styles.nextBtn} onPress={handleNextWord}>
                  <Text style={styles.nextBtnText}>다음 단어 →</Text>
                </Pressable>
              </View>
            )}
          </View>
        ) : (
          <View style={styles.roadmapList}>
            <Text style={styles.mainTitle}>{childName}의 발음 여행</Text>
            {lessonInfo.map((lesson, i) => (
              <Pressable key={i} style={styles.stepCard} onPress={() => openLesson(i)}>
                <Text style={styles.stepEmoji}>{lesson.emoji}</Text>
                <View>
                  <Text style={styles.stepTitle}>{lesson.title}</Text>
                  <Text style={styles.stepSubTitle}>{lesson.subtitle}</Text>
                </View>
              </Pressable>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F9FBF4' },
  container: { padding: 20 },
  backButton: { marginBottom: 15 },
  backButtonText: { color: '#63D400', fontWeight: 'bold', fontSize: 16 },
  exerciseCard: { backgroundColor: 'white', borderRadius: 25, padding: 25, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 10, elevation: 5 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 30 },
  badge: { backgroundColor: '#63D400', color: 'white', paddingHorizontal: 12, paddingVertical: 4, borderRadius: 10, fontWeight: 'bold', overflow: 'hidden' },
  progressText: { color: '#999', fontWeight: 'bold' },
  wordArea: { flexDirection: 'row', alignItems: 'center', marginBottom: 40, gap: 20 },
  wordCircle: { width: 80, height: 80, borderRadius: 40, backgroundColor: '#F1FBE7', alignItems: 'center', justifyContent: 'center' },
  wordEmoji: { fontSize: 35 },
  wordText: { fontSize: 42, fontWeight: '900', color: '#333' },
  btnRow: { flexDirection: 'row', gap: 12 },
  subBtn: { flex: 1, backgroundColor: '#EEF9E3', padding: 18, borderRadius: 15, alignItems: 'center' },
  subBtnText: { color: '#468D00', fontWeight: 'bold', fontSize: 16 },
  mainBtn: { flex: 2, backgroundColor: '#63D400', padding: 18, borderRadius: 15, alignItems: 'center' },
  recordingBtn: { backgroundColor: '#FF5C5C' },
  mainBtnText: { color: 'white', fontWeight: 'bold', fontSize: 16 },
  feedbackContainer: { marginTop: 30, borderTopWidth: 1, borderColor: '#F0F0F0', paddingTop: 20, alignItems: 'center' },
  feedbackTitle: { fontSize: 20, fontWeight: 'bold', color: '#333' },
  feedbackSub: { color: '#666', marginVertical: 8 },
  nextBtn: { backgroundColor: '#333', paddingHorizontal: 30, paddingVertical: 12, borderRadius: 12, marginTop: 10 },
  nextBtnText: { color: 'white', fontWeight: 'bold' },
  roadmapList: { gap: 15 },
  mainTitle: { fontSize: 26, fontWeight: '900', color: '#333', marginBottom: 10 },
  stepCard: { flexDirection: 'row', backgroundColor: 'white', padding: 20, borderRadius: 20, alignItems: 'center', gap: 20, borderWidth: 1, borderColor: '#EEE' },
  stepEmoji: { fontSize: 30 },
  stepTitle: { fontSize: 18, fontWeight: 'bold', color: '#333' },
  stepSubTitle: { fontSize: 13, color: '#888', marginTop: 2 }
});