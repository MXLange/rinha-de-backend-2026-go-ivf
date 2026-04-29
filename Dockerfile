FROM golang:1.26.2-alpine3.23 AS build

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY cmd ./cmd
COPY main.go ./
COPY references.json.gz normalization.json mcc_risk.json ./

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /bin/preprocess ./cmd/preprocess
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /bin/test-go-ivf .
RUN mkdir -p /app/resources && /bin/preprocess

FROM alpine:3.23.3

RUN apk add --no-cache ca-certificates wget

WORKDIR /app

COPY --from=build /bin/test-go-ivf /bin/test-go-ivf
COPY --from=build /app/resources /app/resources

ENTRYPOINT ["/bin/test-go-ivf"]
