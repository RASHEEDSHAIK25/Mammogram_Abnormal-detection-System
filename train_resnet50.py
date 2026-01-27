"""
Training script for ResNet50 model.
Two-phase training: feature extraction then fine-tuning.
"""
import os
import tensorflow as tf
from data_utils import get_datasets, CLASS_NAMES
from models import build_resnet50_model

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def main():
    print("=" * 60)
    print("Training ResNet50 Model")
    print("=" * 60)
    
    # Get datasets
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds, _ = get_datasets(batch_size=16)
    
    # Build model
    print("\nBuilding ResNet50 model...")
    model, base_model = build_resnet50_model(num_classes=len(CLASS_NAMES))
    print(f"Model has {model.count_params():,} parameters")
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/resnet50_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Phase 1: Feature extraction (base frozen)
    print("\n" + "=" * 60)
    print("Phase 1: Feature Extraction (Base Frozen)")
    print("=" * 60)
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    print(f"\nPhase 1 - Best val accuracy: {max(history1.history['val_accuracy']):.4f}")
    
    # Phase 2: Fine-tuning (unfreeze last conv block)
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning (Unfreezing last conv block)")
    print("=" * 60)
    
    # Unfreeze last conv block (conv5_block3)
    base_model.trainable = True
    # Freeze all except last block
    for layer in base_model.layers:
        if 'conv5_block3' not in layer.name:
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    # Fine-tuning callbacks
    early_stopping_ft = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint_ft = tf.keras.callbacks.ModelCheckpoint(
        'models/resnet50_best_ft.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping_ft, checkpoint_ft],
        verbose=1
    )
    
    print(f"\nPhase 2 - Best val accuracy: {max(history2.history['val_accuracy']):.4f}")
    
    # Save final model
    model.save('models/resnet50_final.h5')
    print("\nFinal model saved to models/resnet50_final.h5")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Validation Set")
    print("=" * 60)
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()

