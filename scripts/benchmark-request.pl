#!/usr/bin/env perl

use strict;
use warnings;
use File::Temp qw(tempdir);
use Getopt::Long qw(GetOptions);
use IPC::Open3;
use JSON::PP qw(decode_json encode_json);
use Symbol qw(gensym);
use Time::HiRes qw(time);

my $binary = './bliss-playlist-optimizer';
my $command = 'bridge';
my $request;
my $cache_dir;
my $iterations = 3;
GetOptions(
    'binary=s' => \$binary,
    'command=s' => \$command,
    'request=s' => \$request,
    'cache-dir=s' => \$cache_dir,
    'iterations=i' => \$iterations,
) or die "Invalid arguments\n";
die "--request is required\n" unless defined $request && length $request;
die "--command must be route or bridge\n"
    unless $command eq 'route' || $command eq 'bridge';
die "--iterations must be at least 2 to show cold and warm behavior\n"
    unless $iterations >= 2;

$cache_dir ||= tempdir('bliss-playlist-optimizer-benchmark-XXXXXX', TMPDIR => 1, CLEANUP => 1);

for my $iteration (1 .. $iterations) {
    my $stderr = gensym;
    my $started = time();
    my $pid = open3(
        undef,
        my $stdout,
        $stderr,
        $binary,
        $command,
        '--request',
        $request,
        '--timings',
        '--cache-dir',
        $cache_dir,
    );
    local $/;
    my $payload = <$stdout> // '';
    my $error = <$stderr> // '';
    waitpid($pid, 0);
    my $wall_ms = int(1000 * (time() - $started));
    my $exit = $? >> 8;
    die "optimizer failed (exit=$exit): $error\n" if $exit != 0;
    my $artifact = eval { decode_json($payload) }
        or die "optimizer returned invalid JSON: $@\n";
    my $performance = $artifact->{performance}
        or die "optimizer result has no performance object\n";
    print encode_json({
        iteration => $iteration,
        wall_ms => $wall_ms,
        native_total_ms => $performance->{total_ms},
        database_cache => $performance->{database_cache},
        stages => $performance->{stages},
    }), "\n";
}

