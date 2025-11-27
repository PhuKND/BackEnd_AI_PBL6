package com.khanhlms.medical_store.utills;
import java.lang.reflect.Field;

public class ReflexUtills {
    public static<T> void mergeNonNullFields(T taget, T source) {
        if(taget == null || source == null) {
            throw new IllegalArgumentException("Argument 'tager' and 'source' must not be null");
        }
        Class<?> clazz = taget.getClass();
        if (!clazz.equals(source.getClass())) {
            throw new IllegalArgumentException("Argument 'tagerClass' and 'sourceClass' must be the same");
        }
        Field fields[] = clazz.getDeclaredFields();
        for (Field field : fields) {
            field.setAccessible(true);
            try {
                Object fieldValue = field.get(source);
                if (fieldValue != null) {
                    field.set(taget, fieldValue);
                }
            } catch (IllegalAccessException e) {
                throw new RuntimeException("Merge value is error" + e.getMessage());
            }
        }

    }
}
