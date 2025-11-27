package com.khanhlms.medical_store.model;

import lombok.AccessLevel;
import lombok.Data;
import lombok.experimental.FieldDefaults;

@Data
@FieldDefaults(level = AccessLevel.PRIVATE)
public class RegisterModel {
    String testCaseID;
    String testCaseDescription;
    String username;
    String password;
    String repeat;
    Integer expectedStatus;
    String expectedMessage;
}
