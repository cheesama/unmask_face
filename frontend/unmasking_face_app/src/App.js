import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';

let camera: Camera;

export default function App() {
    const [hasPermission, setHasPermission] = useState(null);
    const [type, setType] = useState(Camera.Constants.Type.back);
    
    useEffect(() => {
        (async () => {
            const { status } = await Camera.requestPermissionsAsync();
            setHasPermission(status === 'granted');
        })();
    }, []);

    if (hasPermission === null) {
        return <View />;
    }

    if (hasPermission === false) {
        return <Text>Please enable the camera permission in the app's settings</Text>;
    }    

    return (
        <View style={styles.container}>
            <Camera 
                sytle={styles.camera} 
                type={type}
                ref={(r) => {
                    camera = r
                }}
            >
                <View style={styles.buttonContainer}>
                    <TouchableOpacity 
                        style={styles.button}
                        onPress={() => {
                            setType(
                                type === Camera.Constants.Type.back ? Camera.Constants.Type.front : Camera.Constants.Type.back
                            );
                        }}
                    >
                        
                        <Text style={styles.text}> Flip </Text>
                        <Text 
                            style={styles.text_right}
                            onPress={async () => {
                                if(!camera) return;
                                const photo = await camera.takePictureAsync();
                                console.log(photo);
                            }}
                        >
                            Shot 
                        </Text>
                    </TouchableOpacity>
                </View>
            </Camera>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
      flex: 1,
    },
    camera: {
      flex: 1,
    },
    buttonContainer: {
      flex: 1,
      backgroundColor: 'transparent',
      flexDirection: 'row',
      margin: 20,
    },
    button: {
      flex: 0.1,
      alignSelf: 'flex-end',
      alignItems: 'center',
    },
    text: {
      fontSize: 40,
      color: 'black',
    },
    text_right: {
        fontSize: 60,
        color: 'black',
        position: 'absolute',
        right: -800,
    }
});