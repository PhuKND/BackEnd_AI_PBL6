package com.khanhlms.medical_store.dtos.response.auth;

import lombok.Builder;

import lombok.AllArgsConstructor;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;


@Builder
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class LogoutResponse {
    boolean success;
}
