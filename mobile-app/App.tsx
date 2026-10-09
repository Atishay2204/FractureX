import { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";

type SelectedImage = {
  uri: string;
  name: string;
  type: string;
};

type Analysis = {
  detections: { region: string; confidence: number }[];
  annotated_image: string;
  elapsed_seconds: number;
  disclaimer: string;
};

const API_URL = (process.env.EXPO_PUBLIC_API_URL ?? "http://10.0.2.2:8000").replace(/\/$/, "");

export default function App() {
  const [image, setImage] = useState<SelectedImage | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);

  const chooseImage = async (source: "camera" | "library") => {
    const permission = source === "camera"
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Permission needed", `Allow access to the ${source} to select an X-ray.`);
      return;
    }

    const options: ImagePicker.ImagePickerOptions = {
      mediaTypes: ["images"],
      quality: 0.9,
      allowsEditing: false,
    };
    const result = source === "camera"
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync(options);

    if (result.canceled) return;
    const asset = result.assets[0];
    if (asset.fileSize && asset.fileSize > 10 * 1024 * 1024) {
      Alert.alert("Image too large", "Choose an X-ray image that is 10 MB or smaller.");
      return;
    }
    setImage({
      uri: asset.uri,
      name: asset.fileName ?? `xray-${Date.now()}.jpg`,
      type: asset.mimeType ?? "image/jpeg",
    });
    setAnalysis(null);
  };

  const analyzeImage = async () => {
    if (!image) return;
    setLoading(true);
    setAnalysis(null);
    try {
      const form = new FormData();
      form.append("image", {
        uri: image.uri,
        name: image.name,
        type: image.type,
      } as unknown as Blob);
      form.append("confidence", "5");

      const response = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "The X-ray could not be analyzed.");
      }
      setAnalysis(payload as Analysis);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not reach the analysis server.";
      Alert.alert("Analysis unavailable", `${message}\n\nCheck that the API is running and EXPO_PUBLIC_API_URL points to it.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.logo}>🩻</Text>
          <Text style={styles.brand}>Osteo<Text style={styles.accent}>Scan</Text> AI</Text>
          <View style={styles.status}><View style={styles.dot} /><Text style={styles.statusText}>SCREENING</Text></View>
        </View>

        <View style={styles.hero}>
          <Text style={styles.eyebrow}>X-RAY FRACTURE SCREENING</Text>
          <Text style={styles.title}>A clearer next step.</Text>
          <Text style={styles.subtitle}>Capture an X-ray or choose one from your library to highlight possible areas of concern.</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Your X-ray</Text>
          <Text style={styles.helper}>JPG or PNG · up to 10 MB · processed for analysis</Text>
          <View style={styles.actions}>
            <Pressable accessibilityRole="button" style={styles.primaryButton} onPress={() => chooseImage("camera")}>
              <Text style={styles.primaryText}>Take a photo</Text>
            </Pressable>
            <Pressable accessibilityRole="button" style={styles.secondaryButton} onPress={() => chooseImage("library")}>
              <Text style={styles.secondaryText}>Choose image</Text>
            </Pressable>
          </View>
          {image ? <Image source={{ uri: image.uri }} style={styles.preview} resizeMode="contain" /> : (
            <View style={styles.placeholder}><Text style={styles.placeholderIcon}>＋</Text><Text style={styles.placeholderText}>Your X-ray preview will appear here</Text></View>
          )}
          {image && (
            <Pressable accessibilityRole="button" disabled={loading} style={[styles.primaryButton, styles.analyzeButton]} onPress={analyzeImage}>
              {loading ? <ActivityIndicator color="#062126" /> : <Text style={styles.primaryText}>Analyze X-ray</Text>}
            </Pressable>
          )}
        </View>

        {analysis && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Analysis results</Text>
            <Text style={styles.helper}>AI screening result · review with a clinician</Text>
            <Image source={{ uri: `data:image/png;base64,${analysis.annotated_image}` }} style={styles.preview} resizeMode="contain" />
            {analysis.detections.length === 0 ? (
              <Text style={styles.resultText}>No finding above the selected threshold. This does not rule out a fracture.</Text>
            ) : analysis.detections.map((item, index) => (
              <View key={`${item.region}-${index}`} style={styles.finding}>
                <Text style={styles.findingRegion}>{item.region}</Text>
                <Text style={styles.findingConfidence}>{item.confidence.toFixed(1)}%</Text>
              </View>
            ))}
            <Text style={styles.disclaimer}>{analysis.disclaimer} Show the original X-ray to a doctor or radiologist.</Text>
          </View>
        )}

        <Text style={styles.footer}>For education and demonstration only. Do not use this app to make treatment decisions.</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#060a10" },
  content: { padding: 20, paddingBottom: 36, maxWidth: 720, width: "100%", alignSelf: "center" },
  header: { flexDirection: "row", alignItems: "center", marginBottom: 28 },
  logo: { fontSize: 25, marginRight: 10 },
  brand: { color: "#e0ecf7", fontSize: 20, fontWeight: "700", flex: 1 },
  accent: { color: "#3ecfda" },
  status: { flexDirection: "row", alignItems: "center", gap: 7, backgroundColor: "#10252a", paddingHorizontal: 10, paddingVertical: 7, borderRadius: 20 },
  dot: { width: 7, height: 7, borderRadius: 4, backgroundColor: "#3ecfda" },
  statusText: { color: "#8babb9", fontSize: 10, fontWeight: "700", letterSpacing: 1 },
  hero: { backgroundColor: "#101b28", padding: 22, borderRadius: 22, marginBottom: 18, borderWidth: 1, borderColor: "#203346" },
  eyebrow: { color: "#3ecfda", fontSize: 11, letterSpacing: 1.5, fontWeight: "700", marginBottom: 10 },
  title: { color: "#e0ecf7", fontSize: 30, fontWeight: "700", marginBottom: 8 },
  subtitle: { color: "#a8bfd0", fontSize: 15, lineHeight: 23 },
  card: { backgroundColor: "#0b1219", padding: 18, borderRadius: 20, borderWidth: 1, borderColor: "#203346", marginBottom: 16 },
  sectionTitle: { color: "#e0ecf7", fontSize: 19, fontWeight: "700" },
  helper: { color: "#7590a6", fontSize: 12, lineHeight: 18, marginTop: 5, marginBottom: 16 },
  actions: { gap: 10, marginBottom: 14 },
  primaryButton: { minHeight: 50, borderRadius: 12, alignItems: "center", justifyContent: "center", backgroundColor: "#3ecfda", paddingHorizontal: 16 },
  primaryText: { color: "#062126", fontSize: 15, fontWeight: "700" },
  secondaryButton: { minHeight: 48, borderRadius: 12, alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: "#315369", paddingHorizontal: 16 },
  secondaryText: { color: "#b7d1df", fontSize: 14, fontWeight: "600" },
  preview: { width: "100%", height: 280, backgroundColor: "#060a10", borderRadius: 12, marginTop: 4, marginBottom: 14 },
  placeholder: { height: 150, backgroundColor: "#101b28", borderRadius: 12, alignItems: "center", justifyContent: "center", marginBottom: 14 },
  placeholderIcon: { color: "#3ecfda", fontSize: 32, marginBottom: 6 },
  placeholderText: { color: "#7590a6", fontSize: 13 },
  analyzeButton: { marginTop: 2 },
  resultText: { color: "#f5c975", fontSize: 14, lineHeight: 21, marginBottom: 10 },
  finding: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 13, borderBottomWidth: 1, borderBottomColor: "#203346" },
  findingRegion: { color: "#e0ecf7", fontSize: 14, fontWeight: "600" },
  findingConfidence: { color: "#3ecfda", fontSize: 14, fontWeight: "700" },
  disclaimer: { color: "#9ab0bf", fontSize: 13, lineHeight: 20, marginTop: 16 },
  footer: { color: "#657f94", fontSize: 12, lineHeight: 18, textAlign: "center", paddingHorizontal: 12, marginTop: 8 },
});
