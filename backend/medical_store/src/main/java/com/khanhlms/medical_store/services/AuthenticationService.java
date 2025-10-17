package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.requests.auth.LoginRequest;
import com.khanhlms.medical_store.dtos.requests.auth.LogoutRequest;
import com.khanhlms.medical_store.dtos.response.auth.LoginResponse;
import com.khanhlms.medical_store.dtos.response.auth.LogoutResponse;
import com.khanhlms.medical_store.entities.RoleEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.repositories.UserRepository;
import com.khanhlms.medical_store.utills.BaseRedisUtils;
import com.nimbusds.jose.*;
import com.nimbusds.jose.crypto.MACSigner;
import com.nimbusds.jose.crypto.MACVerifier;
import com.nimbusds.jwt.JWTClaimsSet;
import com.nimbusds.jwt.SignedJWT;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.stereotype.Service;

import java.text.ParseException;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class AuthenticationService {
    final AuthenticationManager authenticationManager;
    final UserRepository userRepository;
    final BaseRedisUtils baseRedisUtils;

    private static final String TOKEN_PREFIX = "BLACKLIST:TOKEN:";

    @Value("${app.security.expiresIn}")
    private Integer expiresIn;
    @Value("${app.security.expiresRefresh}")
    private Integer expiresRefresh;
    @Value("${app.security.secret}")
    private String SECRET;


    public LoginResponse hanlelogin(LoginRequest loginRequest) {
        UsernamePasswordAuthenticationToken token = new UsernamePasswordAuthenticationToken(loginRequest.getUsername(), loginRequest.getPassword());
        authenticationManager.authenticate(token);
        return LoginResponse.builder()
                .tokenType("Bearer")
                .accessToken(generateToken(loginRequest.getUsername(), true))
                .expiresIn(expiresIn)
                .refreshToken(generateToken(loginRequest.getUsername(), false))
                .build();

    }

    public String generateToken(String username, boolean isAccessToken) {
        Date now = new Date();
        long durationInMillis = isAccessToken ? expiresIn * 1000 : expiresRefresh * 1000;
        Date expiration = new Date(now.getTime() + durationInMillis);

        JWSHeader header = new JWSHeader(JWSAlgorithm.HS512);
        JWTClaimsSet.Builder builder = new JWTClaimsSet.Builder()
                .issuer("medical_store")
                .subject(username)
                .jwtID(UUID.randomUUID().toString())
                .issueTime(now)
                .expirationTime(expiration);
        if (isAccessToken) {
            UserEntity user = this.userRepository.findByUsername(username)
                    .orElseThrow(() -> new AppException(ErrorCode.AUTHENTICATION_EXCEPTION));
            Set<String> roles = getRole(user.getRoles());
            builder.claim("token-type", "access")
                    .claim("scope", buildScope(roles));
        } else {
            builder.claim("token-type", "refresh");
        }
        SignedJWT signedJWT = new SignedJWT(header, builder.build());
        try {
            JWSSigner jwsSigner = new MACSigner(SECRET.getBytes());
            signedJWT.sign(jwsSigner);
        } catch (JOSEException e) {
            throw new RuntimeException("ERROR SIGNATURE TOKEN! ");
        }

        return signedJWT.serialize();
    }

    public boolean introspectToken(String token) {
        try {
            SignedJWT signedJWT = SignedJWT.parse(token);

            boolean verified = signedJWT.verify(new MACVerifier(SECRET.getBytes()));

            JWTClaimsSet  claimsSet = signedJWT.getJWTClaimsSet();
            String jwtID = claimsSet.getJWTID();
            Date expires = claimsSet.getExpirationTime();
            Date now = new Date();
            boolean checkBlacllist = checkTokenInBlackList(jwtID);
            if (expires == null || verified == false || now.after(expires) || checkBlacllist == true) {
                return false;
            }
            return true;
        } catch (ParseException e) {
            throw new RuntimeException(e);
        } catch (JOSEException e) {
            throw new RuntimeException(e);
        }
    }
    private String buildScope(Set<String> roles) {
        StringJoiner joiner = new StringJoiner(" ", "", "");
        for (String item : roles) {
            joiner.add(item);
        }
        return joiner.toString();
    }

    private Set<String> getRole(Set<RoleEntity> entities) {
        Set<String> result = new HashSet<>();
        for (RoleEntity roleEntity : entities) {
            result.add(roleEntity.getName());
        }
        return result;
    }
    /*
        Finished Auth
    */


    public LogoutResponse handLogout(LogoutRequest logoutRequest) {
        String accessToken = logoutRequest.getAccessToken();
        String refreshToken = logoutRequest.getRefreshToken();
        this.baseRedisUtils.set(this.TOKEN_PREFIX + this.jwtID(accessToken),true, this.timeToLongToken(accessToken), TimeUnit.SECONDS);
        this.baseRedisUtils.set(this.TOKEN_PREFIX + this.jwtID(refreshToken),true, this.timeToLongToken(refreshToken), TimeUnit.SECONDS);

        return LogoutResponse.builder()
                .success(true)
                .build();
    }
    private long timeToLongToken(String token) {
        SignedJWT signedJWT = null;
        try {
            signedJWT = SignedJWT.parse(token);
            JWTClaimsSet jwtClaimsSet = signedJWT.getJWTClaimsSet();
            Date expirationTime = jwtClaimsSet.getExpirationTime();
            Date now = new Date();
            long durationInMillis = expirationTime.getTime() - now.getTime();
            return durationInMillis / 1000;
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }
    private String jwtID (String token) {
        try {
            SignedJWT signedJWT = SignedJWT.parse(token);
            JWTClaimsSet jwtClaimsSet = signedJWT.getJWTClaimsSet();
            return  jwtClaimsSet.getJWTID();
        }
        catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }
    private boolean checkTokenInBlackList(String jwtID) {
        Object result = this.baseRedisUtils.getForString(this.TOKEN_PREFIX+jwtID);
        return  Boolean.TRUE.equals(result);
    }
}

