#! /usr/local/bin/perl

use lib '/home/chhuang/clairlib/lib';
use Clair::Utils::TFIDFUtils;
use Clair::Utils::CorpusDownload;
use Lingua::Stem;
use strict;

our $train_folder;
our $test_folder;
our $output;
our $m;
our $feature_output;
our %vocabulary;
our $counter = 1;
our @voc_invert;

# output features with TF*IDF values.
if ($#ARGV != 2 and $#ARGV != 4) {
    usage();
    die;
}

$train_folder = shift;
$test_folder = shift;
$output = shift;
if ($#ARGV == 4) {
    $m = shift;
    $feature_output = shift;
}

getTrainData();
getTestData();

# output the vocabulary
open (FOUT, ">./vocabulary") || die $!;

for (my $i = 1; $i <= $counter; $i++) {
    print FOUT "$i $voc_invert[$i]\n";
}

close(FOUT);

sub getTestData {
# Do the same with the test data
# get the number of classes
    my $classes = `ls $test_folder | wc -l`;
    chomp($classes);

# process the training data first
# for each class, tokenization all texts within the training data
    my @folders = `ls $test_folder`;
    my $no_folders = `find $test_folder -type f | wc -l`;
    chomp($no_folders);
    open (FOUT, ">./$output.test.tmp") || die $!;
    print FOUT "#class:$classes docs:$no_folders\n";
    foreach my $folder(@folders) {
        my $class = $folder;
        chomp($class);
        $folder = $test_folder.'/'.$folder;
        chomp($folder);

        my @files = `ls $folder`;
        foreach my $file(@files) {
            chomp($file);
            my %TFs;
            # slurp the content
            open(FIN, "<$folder/$file") || die $!;
            my @lines = <FIN>;
            close(FIN);

            #split into words
            my @words = split_words(join(" ",@lines)); 

            #stem it
            my $stemmer = Lingua::Stem->new(-locale => 'EN-UK');
            $stemmer->stem_caching({ -level => 2 });
            my $stemmmed_words_anon_array   = $stemmer->stem_in_place(@words);

            # build vocabulary & TF
            for my $word(@$stemmmed_words_anon_array) {
                if (! exists $vocabulary{$word}) {
                    $vocabulary{$word} = $counter;
                    $voc_invert[$counter++] = $word;
                }
                $TFs{$word} += 1;
            }

            # print out the TF table
            print FOUT "$class";
            for (my $i = 1; $i <= $counter; $i++) {
                if (exists $TFs{$voc_invert[$i]}) {
                    print FOUT " $i:$TFs{$voc_invert[$i]}";
                }
            }
            print FOUT " # $file\n";
        }
    }

    close(FOUT);
}

sub getTrainData {
# get the number of classes
    my $classes = `ls $train_folder | wc -l`;
    chomp($classes);

# process the training data first
# for each class, tokenization all texts within the training data
    my @folders = `ls $train_folder`;
    my $no_folders = `find $train_folder -type f | wc -l`;
    chomp($no_folders);
    open (FOUT, ">./$output.train.tmp") || die $!;
    print FOUT "#class:$classes docs:$no_folders\n";
    foreach my $folder(@folders) {
        my $class = $folder;
        chomp($class);
        $folder = $train_folder.'/'.$folder;
        chomp($folder);

        my @files = `ls $folder`;
        foreach my $file(@files) {
            my %TFs;
            # slurp the content
            open(FIN, "<$folder/$file") || die $!;
            my @lines = <FIN>;
            close(FIN);

            #split into words
            my @words = split_words(join(" ",@lines)); 

            #stem it
            my $stemmer = Lingua::Stem->new(-locale => 'EN-UK');
            $stemmer->stem_caching({ -level => 2 });
            my $stemmmed_words_anon_array   = $stemmer->stem_in_place(@words);

            # build vocabulary & TF
            for my $word(@$stemmmed_words_anon_array) {
                if (! exists $vocabulary{$word}) {
                    $vocabulary{$word} = $counter;
                    $voc_invert[$counter++] = $word;
                }
                $TFs{$word} += 1;
            }

            # print out the TF table
            print FOUT "$class";
            for (my $i = 1; $i <= $counter; $i++) {
                if (exists $TFs{$voc_invert[$i]}) {
                    print FOUT " $i:$TFs{$voc_invert[$i]}";
                }
            }
            print FOUT "\n";
        }
    }

    close(FOUT);
}

sub usage {
    print "
    Usage: ./preprocess.pl train_dir test_dir output [M feature_file]\n
    -train_dir – Path to documents in the training set.\n
    -test_dir – Path to documents in the testing set.\n
    -output – You should produce labeled feature vectors in output.train and output.test for
    the training and test sets, respectively.
    - M– (optional) number of features to use for chi-square feature selection.\n
    - feature_file– (optional) When feature selection is used, there is also another\n
    parameter, feature_file. You should output the chi-square scores for all words to this
    file (e.g., each line should have: word <tab> score).\n
    ";
}
